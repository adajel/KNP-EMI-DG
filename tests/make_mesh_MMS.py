#!/usr/bin/env python3

"""
This script generates a 2D unit square mesh with a cell at [0.25,
0.75]x[0.25, 0.75] for MMS test
"""

from dolfin import *
import sys


class Top(SubDomain):
    # define exterior boundary
    def inside(self, x, on_boundary):
        return near(x[1], 1) and on_boundary


class Bottom(SubDomain):
    # define exterior boundary
    def inside(self, x, on_boundary):
        return near(x[1], 0) and on_boundary


class LeftSide(SubDomain):
    # define exterior boundary
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary


class RightSide(SubDomain):
    # define exterior boundary
    def inside(self, x, on_boundary):
        return near(x[0], 1) and on_boundary


import argparse
from pathlib import Path


class CustomParser(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
): ...


def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=CustomParser)
    parser.add_argument(
        "-r",
        "--resolution",
        dest="resolution_factor",
        default=4,
        type=int,
        help="Mesh resolution factor",
    )
    parser.add_argument(
        "-d",
        "--directory",
        dest="mesh_dir",
        type=Path,
        default=Path("meshes/MMS"),
        help="Directory to save the mesh",
    )
    args = parser.parse_args(argv)
    resolution_factor = args.resolution_factor
    out_dir = args.mesh_dir
    nx = 2**resolution_factor
    ny = 2**resolution_factor

    # box mesh
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)

    a = Point(0.25, 0.25)  # bottom left of interior domain
    b = Point(0.75, 0.75)  # top right of interior domain

    # define interior domain
    in_interior = """ (x[0] >= %g && x[0] <= %g &&
                    x[1] >= %g && x[1] <= %g)""" % (a[0], b[0], a[1], b[1])

    interior = CompiledSubDomain(in_interior)

    # mark interior (1) and exterior domain (0)
    # subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 2)
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for cell in cells(mesh):
        x = cell.midpoint().array()
        if interior.inside(x, False):
            subdomains[cell] = 1
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    # mark interface facets / surfaces of mesh
    surfaces = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        side_1 = near(x[0], a[0]) and a[1] <= x[1] <= b[1]
        side_2 = near(x[1], a[1]) and a[0] <= x[0] <= b[0]
        side_3 = near(x[0], b[0]) and a[1] <= x[1] <= b[1]
        side_4 = near(x[1], b[1]) and a[0] <= x[0] <= b[0]
        surfaces[facet] = side_1 * 1 or side_2 * 2 or side_3 * 3 or side_4 * 4

    # mark exterior boundary
    LeftSide().mark(surfaces, 5)
    Bottom().mark(surfaces, 6)
    RightSide().mark(surfaces, 7)
    Top().mark(surfaces, 8)

    # save .xml files
    mesh_file = File(str((out_dir / f"mesh_{resolution_factor}.xml").absolute()))
    mesh_file << mesh

    subdomains_file = File(
        str((out_dir / f"subdomains_{resolution_factor}.xml").absolute())
    )
    subdomains_file << subdomains

    surfaces_file = File(
        str((out_dir / f"surfaces_{resolution_factor}.xml").absolute())
    )
    surfaces_file << surfaces

if __name__ == "__main__":
    main()
