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

# if no input argument, set resolution factor to default = 0
if len(sys.argv) == 1:
    resolution_factor = 4
else:
    resolution_factor = int(sys.argv[1])

nx = 2**resolution_factor
ny = 2**resolution_factor

# box mesh
mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)

a = Point(0.25, 0.25)   # bottom left of interior domain
b = Point(0.75, 0.75)   # top right of interior domain

# define interior domain
in_interior = """ (x[0] >= %g && x[0] <= %g &&
                   x[1] >= %g && x[1] <= %g)""" \
                   % (a[0], b[0], a[1], b[1])

interior = CompiledSubDomain(in_interior)

# mark interior (1) and exterior domain (2)
subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 2)
for cell in cells(mesh):
    x = cell.midpoint().array()
    if interior.inside(x, False):
        subdomains[cell] = 1
assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

# mark interface facets / surfaces of mesh
surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
for facet in facets(mesh):
    x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
    side_1 = (near(x[0], a[0]) and a[1] <= x[1] <= b[1])
    side_2 = (near(x[1], a[1]) and a[0] <= x[0] <= b[0])
    side_3 = (near(x[0], b[0]) and a[1] <= x[1] <= b[1])
    side_4 = (near(x[1], b[1]) and a[0] <= x[0] <= b[0])
    surfaces[facet] = side_1*1 or side_2*2 or side_3*3 or side_4*4

# mark exterior boundary
LeftSide().mark(surfaces, 5)
Bottom().mark(surfaces, 6)
RightSide().mark(surfaces, 7)
Top().mark(surfaces, 8)

# save .xml files
mesh_file = File('meshes/MMS/mesh_' + str(resolution_factor) + '.xml')
mesh_file << mesh

subdomains_file = File('meshes/MMS/subdomains_' + str(resolution_factor) + '.xml')
subdomains_file << subdomains

surfaces_file = File('meshes/MMS/surfaces_' + str(resolution_factor) + '.xml')
surfaces_file << surfaces

# save .pvd files
meshplot = File('meshes/MMS/mesh_' + str(resolution_factor) + '.pvd')
meshplot << subdomains

surfacesplot = File('meshes/MMS/surfaces_' + str(resolution_factor) + '.pvd')
surfacesplot << surfaces
