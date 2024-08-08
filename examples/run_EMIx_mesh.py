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
    Tstop = 1.0e-3                   # global end time (s)
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

    # Set physical parameters
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

    # Create ion list. NB! The last ion in list will be eliminated, and
    # should be the ion with the smallest diffusion coefficient D_k
    ion_list = [K, Cl, Na]

    # Membrane parameters
    g_syn_bar = 0                     # synaptic conductivity (S/m**2)

    # Set stimulus ODE
    stimulus = {'stim_amplitude': g_syn_bar}
    stimulus_locator = lambda x: (x[0] < 0.1e-6)

    stim_params = namedtuple('membrane_params', ('g_syn_bar', \
                             'stimulus', 'stimulus_locator'))(g_syn_bar, \
                              stimulus, stimulus_locator)

    # Get mesh, subdomains, surfaces paths
    mesh_path = 'meshes/emix_meshes/volume_ncells_5_size_5000/'

    mesh = Mesh()
    infile = XDMFFile(mesh_path + 'mesh.xdmf')
    infile.read(mesh)
    cdim = mesh.topology().dim()
    subdomains = MeshFunction("size_t", mesh, cdim)
    infile.read(subdomains, "label")
    infile.close()

    print(np.unique(subdomains.array()))

    # Remark subdomains
    for cell in cells(mesh):
        if subdomains[cell] == 1:
            subdomains[cell] = 0
        else:
            subdomains[cell] = 1

    print(np.unique(subdomains.array()))

    # get all local labels
    File("meshes/emix_meshes/subdomains.pvd") << subdomains

    infile = XDMFFile(mesh_path + 'facets.xdmf')
    surfaces = MeshFunction("size_t", mesh, cdim - 1)
    infile.read(surfaces, "boundaries")
    infile.close()

    print(np.unique(surfaces.array()))

    unique, counts = np.unique(surfaces.array(), return_counts=True)

    #print(dict(zip(unique, counts)))

    int_fac_mem = 0
    int_fac_int = 1
    # Remark facets
    for facet in facets(mesh):
        if surfaces[facet] > 10:
            surfaces[facet] = 5
        elif surfaces[facet] > 0:
            surfaces[facet] = 1
            int_fac_mem +=1
        elif surfaces[facet] == 0:
            int_fac_int +=1

    print(np.unique(surfaces.array()))


    #print(int_fac_int)
    #print(int_fac_mem)
    #print(int_fac_mem/len(surfaces)*100)

    File("meshes/emix_meshes/surfaces.pvd") << surfaces

    # convert mesh to unit meter (m)
    mesh.coordinates()[:,:] *= 1e-9

    #sys.exit(0)

    # Set solver parameters EMI (True is direct, and False is iterate) 
    direct_emi = True
    rtol_emi = 1E-5
    atol_emi = 1E-40
    threshold_emi = None

    # Set solver parameters KNP (True is direct, and False is iterate) 
    direct_knp = True
    rtol_knp = 1E-7
    atol_knp = 1E-40
    threshold_knp = None

    # Set parameters
    solver_params = namedtuple('solver_params', ('direct_emi',
                               'direct_knp', 'resolution',
                               'rtol_emi', 'rtol_knp',
                               'atol_emi', 'atol_knp',
                               'threshold_emi', 'threshold_knp'
                               ))(direct_emi, direct_knp, resolution, \
                                  rtol_emi, rtol_knp, atol_emi, atol_knp, \
                                  threshold_emi, threshold_knp)

    # File for results
    fname = "results/data/3D/"

    # Dictionary with membrane models (key is facet tag, value is ode model)
    ode_models = {1: mm_leak}

    # Solve system
    S = Solver(params, ion_list)                    # create solver
    S.setup_domain(mesh, subdomains, surfaces)      # setup meshes
    S.setup_parameters()                            # setup physical parameters
    S.setup_FEM_spaces()                            # setup function spaces and numerical parameters
    S.setup_membrane_model(stim_params, ode_models) # setup membrane model(s)
    S.solve_system_active(Tstop, t, solver_params, filename=fname) # solve
