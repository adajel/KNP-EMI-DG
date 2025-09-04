#!/usr/bin/env python3

"""
This script generates a 2-D mesh consisting of a box with an embedded neuron
(a smaller box). The dimensions are modified from
https://www.frontiersin.org/articles/10.3389/fncom.2017.00027/full
to make a coarser mesh. The outer block as lengths
Lx = 120 um
Ly = 120 um
while the inner block has lengths
lx = 60 um
ly = 6 um
"""

from dolfin import *
import sys

class Boundary(SubDomain):
    # define exterior boundary
    def inside(self, x, on_boundary):
        return on_boundary

def add_rectangle(mesh, subdomains, surfaces, a, b):
    # define interior domain
    in_interior = """ (x[0] >= %g && x[0] <= %g &&
    x[1] >= %g && x[1] <= %g)""" \
    % (a[0], b[0], a[1], b[1])

    interior = CompiledSubDomain(in_interior)

    # mark interior and exterior domain
    for cell in cells(mesh):
        x = cell.midpoint().array()
        if (int(interior.inside(x, False))):
            subdomains[cell] = 1
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        side_1 = (near(x[0], a[0]) and a[1] <= x[1] <= b[1])
        side_2 = (near(x[0], b[0]) and a[1] <= x[1] <= b[1])
        side_3 = (near(x[1], a[1]) and a[0] <= x[0] <= b[0])
        side_4 = (near(x[1], b[1]) and a[0] <= x[0] <= b[0])
        surfaces[facet] += side_1 or side_2 or side_3 or side_4

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
        default=Path("meshes/2D"),
        help="Directory to save the mesh",
    )
    args = parser.parse_args(argv)
    resolution_factor = args.resolution_factor
    out_dir = args.mesh_dir

    nx = 31*2**resolution_factor
    ny = 2*2**resolution_factor

    # box mesh
    mesh = RectangleMesh(Point(0, 0), Point(62, 4), nx, ny, "crossed")
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)

    # add interior domains (cells)
    a = Point(1, 1)    # bottom left of interior domain
    b = Point(61, 3)   # top right of interior domain
    add_rectangle(mesh, subdomains, surfaces, a, b)

    # mark exterior boundary
    Boundary().mark(surfaces, 5)

    # convert mesh to unit meter (m)
    mesh.coordinates()[:,:] *= 1e-6

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
