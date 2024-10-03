#!/usr/bin/python3

import os
import sys
import time

from dolfin import *
import numpy as np

from collections import namedtuple

from knpemidg import Solver
import mm_hh as mm_hh
import mm_hh_syn as mm_hh_syn
import mm_glial as mm_glial

# Define colors for printing

"""
Use repo emimesh to create mesh based on position, how large the bounding box
should be and how many cells to include. Use the configuration file

    synapse.yml

Some of the data is marked / annotated with synapses. Try to figure out what
the segmentation id (the one displayed in neuroglancer) is for the cells in the
mesh (should be somewhere).

Then, we can use the tool MicronsBinder and caveclient to lookup the position
for the synapses for the cells in the mesh (via the segmentation id). Activate
environment 

    conda activate cave

and then run

    python3 synapse_script.py

"""

if __name__ == "__main__":

    # Set stimulus ODE
    box_width = 5000e-7 # convert from nm to cm
    grid_size = 5

    """ split domain into grid of size grid_size*grid_size*grid_size. Make
        checkerboard pattern and let 1/2 of each white square be an area for a synapse
    ____________
    |_  |_  |_  |
    |s|_|s|_|s|_|
    |_  |_  |_  |
    |s|_|s|_|s|_|
    |_  |_  |_  |
    |s|_|s|_|s|_|
    """

    stim_box_width = box_width/(grid_size*2.0)
    synapse_grid = ""

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                x = i*box_width/grid_size
                y = j*box_width/grid_size
                z = k*box_width/grid_size

                # if i and j are even
                if (i % 2) == 0:
                    if (j % 2) == 0:
                        if (k % 2) == 0:
                            synapse_grid += "(%E < x[0] < %E)*(%E < x[1] < %E)*(%E < x[2] < %E)" % \
                                             (x, x + stim_box_width, y, y + stim_box_width, \
                                              z, z + stim_box_width)

                            if ((i < grid_size-1) or (j < grid_size-1) or (k < grid_size-1)):
                                synapse_grid += " + "

                # if i and j are odd
                if (i % 2) != 0:
                    if (j % 2) != 0:
                        if (k % 2) != 0:
                            synapse_grid += "(%E < x[0] < %E)*(%E < x[1] < %E)*(%E < x[2] < %E)" % \
                                             (x, x + stim_box_width, y, y + stim_box_width, \
                                              z, z + stim_box_width)

                            if ((i < grid_size-1) or (j < grid_size-1) or (k < grid_size-1)):
                                synapse_grid += " + "

    # Get mesh, subdomains, surfaces paths
    mesh_path = 'meshes/synapse/size+5000/dx+20_ncells+50/envelopsize+18/'

    mesh = Mesh()
    infile = XDMFFile(mesh_path + 'mesh.xdmf')
    infile.read(mesh)
    cdim = mesh.topology().dim()
    subdomains = MeshFunction("size_t", mesh, cdim)
    infile.read(subdomains, "label")
    infile.close()

    print(np.unique(subdomains.array()))
    original_tags_cells = np.unique(subdomains.array()).copy()

    # Remark subdomains
    for cell in cells(mesh):
        if subdomains[cell] == 1:
            subdomains[cell] = 0
        elif subdomains[cell] == 50 or subdomains[cell] == 21:
            subdomains[cell] = 2
        else:
            subdomains[cell] = 1

    print(np.unique(subdomains.array()))

    infile = XDMFFile(mesh_path + 'facets.xdmf')
    surfaces = MeshFunction("size_t", mesh, cdim - 1)
    infile.read(surfaces, "boundaries")
    infile.close()

    print(np.unique(surfaces.array()))

    unique, counts = np.unique(surfaces.array(), return_counts=True)

    #print(dict(zip(unique, counts)))
    list_surface_tags = np.unique(surfaces.array())
    list_inner_surface_tags = np.setdiff1d(list_surface_tags, \
                                           original_tags_cells)
    print(list_inner_surface_tags)

    # Remark facets
    for facet in facets(mesh):
        # mark exterior facets
        if (surfaces[facet] in list_inner_surface_tags) and surfaces[facet] != 0:
            surfaces[facet] = max(list_inner_surface_tags)
        # keep facet marking for interior facets
        elif surfaces[facet] == 0:
            surfaces[facet] = 0
        # mark neuron facets
        elif surfaces[facet] == 50:
            surfaces[facet] = 2
        # mark glial facets
        else:
            surfaces[facet] = 1

    print(np.unique(surfaces.array()))

    # remark synapse (NB! DEBUG ONLY)
    # -------------------------------------------
    #    # convert mesh from nm to cm
    mesh.coordinates()[:,:] *= 1e-7

    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        # if surface belongs to neuron and is part of synaptic grid
        if eval(synapse_grid) and surfaces[facet] == 1:
            print("TRUE")
            surfaces[facet] = 3

    # -------------------------------------------

    # write to file test
    File(mesh_path + "surfaces.pvd") << surfaces
    File(mesh_path + "subdomains.pvd") << subdomains

    File(mesh_path + "surfaces.xml") << surfaces
    File(mesh_path + "subdomains.xml") << subdomains
