#!/usr/bin/python3

import os
import sys
import time
import yaml

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

    stim_box_width = 400 # nm
    synapse_grid = ""

    fname = 'synapse_location_data_10.yml'

    with open(fname, 'r') as file:
        docs = yaml.safe_load_all(file)

        for doc in docs:
            doc = doc

    synapses = []
    for item, key in doc['synapse_location'].items():
        # filter out glial cell tag
        synapses += key

    for synapse in synapses:
        print(synapse)
        x = synapse[0]
        y = synapse[1]
        z = synapse[2]

        synapse_grid += "(%E < x[0] < %E)*(%E < x[1] < %E)*(%E < x[2] < %E)" % \
                         (x, x + stim_box_width, y, y + stim_box_width, \
                          z, z + stim_box_width)

        if synapse != synapses[-1]:
             synapse_grid += " + "

    print(synapse_grid)

    # Get mesh, subdomains, surfaces paths
    mesh_path = 'meshes/synapse/size+5000/dx+20_ncells+10/envelopsize+18/'

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
        elif subdomains[cell] == 10:
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
        elif surfaces[facet] == 10:
            surfaces[facet] = 2
        # mark glial facets
        else:
            surfaces[facet] = 1

    print(np.unique(surfaces.array()))

    # remark synapse (NB! DEBUG ONLY)
    # -------------------------------------------
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        # if surface belongs to neuron and is part of synaptic grid
        if eval(synapse_grid) and surfaces[facet] == 1:
            print("TRUE")
            surfaces[facet] = 3

    # -------------------------------------------
    # convert mesh from nm to cm
    mesh.coordinates()[:,:] *= 1e-7

    # write to file test
    File(mesh_path + "surfaces.pvd") << surfaces
    File(mesh_path + "subdomains.pvd") << subdomains

    File(mesh_path + "surfaces.xml") << surfaces
    File(mesh_path + "subdomains.xml") << subdomains
