#!/usr/bin/python3

import os
import sys
import time

from dolfin import *
import numpy as np

from collections import namedtuple

from solver import Solver
import mm_hh as mm_hh

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
    #for resolution in [0, 1]:
    for resolution in [1]:

        # Time variables PDEs
        dt = 1.0e-4                      # global time step (s)
        #Tstop = 1.0e-1                  # global end time (s)
        Tstop = 1.0e-3                  # global end time (s)
        t = Constant(0.0)                # time constant

        # Time variables ODEs
        n_steps_ODE = 25                 # number of ODE steps

        # Physical parameters (PDEs)
        C_M = 0.02                       # capacitance
        temperature = 300                # temperature (K)
        F = 96485                        # Faraday's constant (C/mol)
        R = 8.314                        # Gas Constant (J/(K*mol))
        D_Na = Constant(1.33e-9)         # diffusion coefficients Na (m/s)
        D_K = Constant(1.96e-9)          # diffusion coefficients K (m/s)
        D_Cl = Constant(2.03e-9)         # diffusion coefficients Cl (m/s)
        psi = F / (R * temperature)      # shorthand
        C_phi = C_M / dt                 # shorthand

        # initial values (PDEs)
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
        g_syn_bar = 40                   # synaptic conductivity (S/m**2)

        # Set stimulus ODE
        stimulus = {'stim_amplitude': g_syn_bar}
        stimulus_locator = lambda x: (x[0] < 20.0e-6)

        stim_params = namedtuple('membrane_params', ('g_syn_bar', \
                                 'stimulus', 'stimulus_locator'))(g_syn_bar, \
                                  stimulus, stimulus_locator)

        # Get mesh, subdomains, surfaces paths
        mesh_prefix = "meshes/3D/"
        mesh = mesh_prefix + "mesh_" + str(resolution) + ".xml"
        subdomains = mesh_prefix + "subdomains_" + str(resolution) + ".xml"
        surfaces = mesh_prefix + "surfaces_" + str(resolution) + ".xml"

        # Generate mesh if it does not exist
        if not os.path.isfile(mesh):
            script = "make_mesh_3D.py"                 # script
            os.system("python " + script + " " + str(resolution)) # run script

        mesh = Mesh(mesh)
        subdomains = MeshFunction('size_t', mesh, subdomains)
        surfaces = MeshFunction('size_t', mesh, surfaces)

        File("results/data/3D/surfaces.pvd") << surfaces
        File("results/data/3D/subdomains.pvd") << subdomains

        # Set solver parameters (True is direct, and False is iterate)
        direct_emi = False
        direct_knp = False

        #direct_emi = True
        #direct_knp = True

        # Set solver parameters
        solver_params = namedtuple('solver_params', ('direct_emi',
            'direct_knp', 'resolution'))(direct_emi, direct_knp, resolution)

        # File for results
        fname = "results/data/3D/"

        # Dictionary with membrane models (key is facet tag, value is ode model)
        ode_models = {1: mm_hh}

        # Solve system
        S = Solver(params, ion_list)                    # create solver
        S.setup_domain(mesh, subdomains, surfaces)      # setup meshes
        S.setup_parameters()                            # setup physical parameters
        S.setup_FEM_spaces()                            # setup function spaces and numerical parameters
        S.setup_membrane_model(stim_params, ode_models) # setup membrane model(s)
        S.solve_system_active(Tstop, t, solver_params, filename=fname) # solve
