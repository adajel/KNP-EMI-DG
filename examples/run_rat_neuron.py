#!/usr/bin/python3

import os
import sys
import time

from dolfin import *
import numpy as np

from collections import namedtuple

from knpemidg import Solver
import mm_hh as mm_hh
import mm_leak as mm_leak

# Define colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":

    # Resolution factor of mesh
    resolution = 0

    # Time variables (PDEs)
    dt = 1.0e-4                      # global time step (s)
    Tstop = 5.0e-4                   # global end time (s)
    #Tstop = 2.0e-1                   # global end time (s)
    t = Constant(0.0)                # time constant

    # Time variables (ODEs)
    n_steps_ODE = 25                 # number of ODE steps

    # Physical parameters
    C_M = 0.02                       # capacitance
    temperature = 300                # temperature (K)
    F = 96485                        # Faraday's constant (C/mol)
    R = 8.314                        # Gas Constant (J/(K*mol))
    D_Na = Constant(1.33e-9)         # diffusion coefficients Na (m/s)
    D_K = Constant(1.96e-9)          # diffusion coefficients K (m/s)
    D_Cl = Constant(2.03e-9)         # diffusion coefficients Cl (m/s)
    psi = F / (R * temperature)      # shorthand
    C_phi = C_M / dt                 # shorthand

    # Initial values
    Na_i_init = Constant(12)         # Intracellular Na concentration
    Na_e_init = Constant(100)        # extracellular Na concentration
    K_i_init = Constant(125)         # intracellular K concentration
    K_e_init = Constant(4)           # extracellular K concentration
    Cl_i_init = Constant(137)        # intracellular Cl concentration
    Cl_e_init = Constant(104)        # extracellular CL concentration
    phi_M_init = Constant(-0.067738) # membrane potential (V)

    # Set parameters
    params = namedtuple('params', ('dt', 'n_steps_ODE', 'F', 'psi', \
            'phi_M_init', 'C_phi', 'C_M', 'R', 'temperature'))(dt, \
            n_steps_ODE, F, psi, phi_M_init, C_phi, C_M, R, temperature)

    # diffusion coefficients for each sub-domain
    D_Na_sub = {1:D_Na, 0:D_Na}
    D_K_sub = {1:D_K, 0:D_K}
    D_Cl_sub = {1:D_Cl, 0:D_Cl}

    # initial concentrations for each sub-domain
    Na_init_sub = {1:Na_i_init, 0:Na_e_init}
    K_init_sub = {1:K_i_init, 0:K_e_init}
    Cl_init_sub = {1:Cl_i_init, 0:Cl_e_init}

    # Create ions (channel conductivity is set below in the membrane model)
    Na = {'c_init_sub':Na_init_sub, 'bdry': Constant((0, 0)),
          'z':1.0, 'name':'Na', 'D_sub':D_Na_sub}
    K = {'c_init_sub':K_init_sub, 'bdry': Constant((0, 0)),
         'z':1.0, 'name':'K', 'D_sub':D_K_sub}
    Cl = {'c_init_sub':Cl_init_sub, 'bdry': Constant((0, 0)),
          'z':-1.0, 'name':'Cl', 'D_sub':D_Cl_sub}

    # Create ion list. NB! The last ion in list will be eliminated, and should 
    # be the ion with the smallest diffusion coefficient
    ion_list = [K, Cl, Na]

    # synaptic conductivity (S/m**2)
    g_syn_bar = 200

    # set stimulus ODE
    stimulus = {'stim_amplitude': g_syn_bar}
    stimulus_locator = lambda x: (x[1] < -80e-6) or (x[0] < -125e-6) or (x[0] > 140e-6)

    stim_params = namedtuple('membrane_params', ('g_syn_bar', \
                             'stimulus', 'stimulus_locator'))(g_syn_bar, \
                              stimulus, stimulus_locator)

    # Get mesh, subdomains, surfaces paths
    file_name = 'meshes/rat_neuron/228-16MG.CNG.xdmf'

    # Create mesh
    mesh = Mesh()
    with XDMFFile(mesh.mpi_comm(), file_name) as xdmf:
        xdmf.read(mesh)

    # define subdomains (ICS and ECS)
    cdim = mesh.topology().dim()
    cell_f = MeshFunction('size_t', mesh, cdim, 0)
    with XDMFFile(mesh.mpi_comm(), file_name) as xdmf:
        xdmf.read(cell_f, 'label')

    # Print for debug
    nzero_cells = np.sum(cell_f.array() == 1)
    none_cells = np.sum(cell_f.array() == 2)
    print('# zeros', nzero_cells, '# ones', none_cells)

    # Remark subdomains
    for cell in cells(mesh):
        if cell_f[cell] == 1:
            cell_f[cell] = 0
        elif cell_f[cell] == 2:
            cell_f[cell] = 1
        else:
            print("no cell found")

    # Print for debug
    nzero_cells = np.sum(cell_f.array() == 1)
    none_cells = np.sum(cell_f.array() == 0)

    print('# zeros', nzero_cells, '# ones', none_cells)

    fdim = cdim-1
    facet_f = MeshFunction('size_t', mesh, fdim, 0)
    DomainBoundary().mark(facet_f, 5)

    # It might pay off to compute the interfaces once and seve them
    mesh.init(fdim, cdim)
    # Since 5 is used for boundary at this point we are only looping over the
    # interior faces
    for f in SubsetIterator(facet_f, 0):
        c0, c1 = f.entities(cdim)
        # Disagreement in tags means the interface
        if cell_f[c0] != cell_f[c1]:
            facet_f[f.index()] = 1
            if f.midpoint().y() >= -5.0:
                facet_f[f.index()] = 2

    # Convert mesh to unit meter (m)
    mesh.coordinates()[:, :] *= 1e-6

    subdomains = cell_f
    surfaces = facet_f

    with XDMFFile('results/data/rat_neuron/subdomains.xdmf') as xdmf:
        xdmf.write(cell_f)

    # Set solver parameters EMI (True is direct, and False is iterate) 
    direct_emi = False
    rtol_emi = 1E-5
    atol_emi = 1E-40
    threshold_emi = 0.9

    # Set solver parameters KNP (True is direct, and False is iterate) 
    direct_knp = False
    rtol_knp = 1E-7
    atol_knp = 2E-40
    threshold_knp = 0.75

    # Set parameters
    solver_params = namedtuple('solver_params', ('direct_emi',
                               'direct_knp', 'resolution',
                               'rtol_emi', 'rtol_knp',
                               'atol_emi', 'atol_knp',
                               'threshold_emi', 'threshold_knp'
                               ))(direct_emi, direct_knp, resolution, \
                                  rtol_emi, rtol_knp, atol_emi, atol_knp, \
                                  threshold_emi, threshold_knp)

    # File for saving results
    fname = "results/data/rat_neuron/"

    # Dictionary with membrane models (key is facet tag, value is ode model)
    ode_models = {1: mm_leak, 2: mm_hh}

    # Solve system
    S = Solver(params, ion_list)                    # create solver
    S.setup_domain(mesh, subdomains, surfaces)      # setup meshes
    S.setup_parameters()                            # setup physical parameters
    S.setup_FEM_spaces()                            # setup function spaces and numerical parameters
    S.setup_membrane_model(stim_params, ode_models) # setup membrane model(s)
    S.solve_system_active(Tstop, t, solver_params, filename=fname) # solve
