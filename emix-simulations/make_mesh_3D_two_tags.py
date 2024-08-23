#!/usr/bin/env python3

"""
This script generates a 3D mesh representing 4 axons.
"""

from dolfin import *
import sys

class Boundary(SubDomain):
    # define exterior boundary
    def inside(self, x, on_boundary):
        return on_boundary

def add_axon(mesh, subdomains, surfaces, a, b, tag):
    # define interior domain
    in_interior = """ (x[0] >= %g && x[0] <= %g &&
                       x[1] >= %g && x[1] <= %g &&
                       x[2] >= %g && x[2] <= %g) """ % (
        a[0],
        b[0],
        a[1],
        b[1],
        a[2],
        b[2],
    )

    interior = CompiledSubDomain(in_interior)

    # mark interior and exterior domain
    for cell in cells(mesh):
        x = cell.midpoint().array()
        if (int(interior.inside(x, False))):
            subdomains[cell] = tag
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        side_1 = near(x[0], a[0]) and a[1] <= x[1] <= b[1] and a[2] <= x[2] <= b[2]
        side_2 = near(x[0], b[0]) and a[1] <= x[1] <= b[1] and a[2] <= x[2] <= b[2]
        side_3 = near(x[1], a[1]) and a[0] <= x[0] <= b[0] and a[2] <= x[2] <= b[2]
        side_4 = near(x[1], b[1]) and a[0] <= x[0] <= b[0] and a[2] <= x[2] <= b[2]
        side_5 = near(x[2], a[2]) and a[0] <= x[0] <= b[0] and a[1] <= x[1] <= b[1]
        side_6 = near(x[2], b[2]) and a[0] <= x[0] <= b[0] and a[1] <= x[1] <= b[1]
        if (side_1 or side_2 or side_3 or side_4 or side_5 or side_6):
            surfaces[facet] = tag

    return

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
        default=0,
        type=int,
        help="Mesh resolution factor",
    )
    parser.add_argument(
        "-d",
        "--directory",
        dest="mesh_dir",
        type=Path,
        default=Path("meshes/3D_two_tags"),
        help="Directory to save the mesh",
    )
    args = parser.parse_args(argv)
    resolution_factor = args.resolution_factor
    out_dir = args.mesh_dir

    l = 1
    nx = l * 20 * 2 ** resolution_factor
    ny = 20 * 2 ** resolution_factor
    nz = 20 * 2 ** resolution_factor

    # box mesh
    mesh = BoxMesh(Point(0, 0.0, 0.0), Point(l * 20, 2.0, 2.0), nx, ny, nz)
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    surfaces = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    a = Point(5, 0.2, 0.2)
    b = Point(l * 20 - 5, 0.6, 0.6)
    add_axon(mesh, subdomains, surfaces, a, b, 1)

    a = Point(5, 0.8, 0.8)
    b = Point(l * 20 - 5, 1.4, 1.4)
    add_axon(mesh, subdomains, surfaces, a, b, 2)

    # mark exterior boundary
    Boundary().mark(surfaces, 5)

    # convert mesh from um to cm
    mesh.coordinates()[:, :] *= 1e-4

    # save .xml files
    mesh_file = File(str((out_dir / f"mesh_{resolution_factor}.xml").absolute()))
    mesh_file << mesh

    subdomains_file = File(str((out_dir / f"subdomains_{resolution_factor}.xml").absolute()))
    subdomains_file << subdomains

    surfaces_file = File(str((out_dir / f"surfaces_{resolution_factor}.xml").absolute()))
    surfaces_file << surfaces

    # save .pvd files
    mesh_file = File(str((out_dir / f"mesh_{resolution_factor}.pvd").absolute()))
    mesh_file << mesh

    subdomains_file = File(str((out_dir / f"subdomains_{resolution_factor}.pvd").absolute()))
    subdomains_file << subdomains

    surfaces_file = File(str((out_dir / f"surfaces_{resolution_factor}.pvd").absolute()))
    surfaces_file << surfaces
