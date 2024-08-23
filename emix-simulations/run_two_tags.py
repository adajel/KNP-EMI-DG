#!/usr/bin/python3

import os
import sys
import time

from dolfin import *
import numpy as np

from collections import namedtuple

from knpemidg import Solver
import mm_hh as mm_hh
import mm_glial as mm_glial
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

    # Time variables PDEs
    dt = 0.1                       # global time step (ms)
    Tstop = 10                     # global end time (ms)
    t = Constant(0.0)              # time constant

    # Time variables ODEs
    n_steps_ODE = 20               # number of ODE steps

    # Physical parameters (PDEs)
    C_M = 2.0                      # capacitance
    temperature = 300e3            # temperature (m K)
    R = 8.314e3                    # Gas Constant (m J/(K mol))
    F = 96485e3                    # Faraday's constant (mC/ mol)

    D_Na = Constant(1.33e-8)       # diffusion coefficients Na (cm/ms)
    D_K = Constant(1.96e-8)        # diffusion coefficients K (cm/ms)
    D_Cl = Constant(2.03e-8)       # diffusion coefficients Cl (cm/ms)

    psi = F / (R * temperature)    # shorthand
    C_phi = C_M / dt               # shorthand

    n = 20
    fname_callibrate = 'results/data/calibration_two_tags/results.h5'
    hdf5file = HDF5File(MPI.comm_world, fname_callibrate, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    P1 = FiniteElement('DG', mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement(2*[P1]))
    V = FunctionSpace(mesh, P1)

    u = Function(W)
    v = Function(V)
    w = Function(V)

    Na_init = Function(V)
    K_init = Function(V)
    Cl_init = Function(V)

    hdf5file.read(u, "/concentrations/vector_" + str(n))

    # K concentrations
    assign(K_init, u.sub(0))
    # Cl concentrations
    assign(Cl_init, u.sub(1))

    # Na concentrations
    hdf5file.read(v, "/elim_concentration/vector_" + str(n))
    assign(Na_init, v)

    hdf5file.close()

    # Set physical parameters
    params = namedtuple('params', ('dt', 'n_steps_ODE', 'F', 'psi', \
                'C_phi', 'C_M', 'R', 'temperature'))(dt, \
                n_steps_ODE, F, psi, C_phi, C_M, R, temperature)

    # diffusion coefficients for each sub-domain
    D_Na_sub = {0:D_Na, 1:D_Na, 2:D_Na}
    D_K_sub = {0:D_K, 1:D_K, 2:D_K}
    D_Cl_sub = {0:D_Cl, 1:D_Cl, 2:D_Cl}

    # initial concentrations for each ion
    Na_init_sub = Na_init
    K_init_sub = K_init
    Cl_init_sub = Cl_init
    c_init_sub_type = 'function' # type - function or constant

    # Create ions (channel conductivity is set below in the membrane model)
    Na = {'c_init_sub':Na_init_sub, 'c_init_sub_type':c_init_sub_type,
          'bdry': Constant((0, 0)), 'z':1.0, 'name':'Na', 'D_sub':D_Na_sub}
    K = {'c_init_sub':K_init_sub, 'c_init_sub_type':c_init_sub_type,
         'bdry': Constant((0, 0)), 'z':1.0, 'name':'K', 'D_sub':D_K_sub}
    Cl = {'c_init_sub':Cl_init_sub, 'c_init_sub_type':c_init_sub_type,
          'bdry': Constant((0, 0)), 'z':-1.0, 'name':'Cl', 'D_sub':D_Cl_sub}

    # Create ion list. NB! The last ion in list will be eliminated, and
    # should be the ion with the smallest diffusion coefficient D_k
    ion_list = [K, Cl, Na]

    # Membrane parameters
    g_syn_bar = 5.0                  # synaptic conductivity (mS/cm**2)
    #g_syn_bar = 0.0                   # synaptic conductivity (mS/cm**2)

    # Set stimulus ODE
    stimulus = {'stim_amplitude': g_syn_bar}
    stimulus_locator = lambda x: (x[0] < 7.0e-4)

    stim_params = namedtuple('membrane_params', ('g_syn_bar', \
                             'stimulus', 'stimulus_locator'))(g_syn_bar, \
                              stimulus, stimulus_locator)

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
                               ))(direct_emi, direct_knp, 0, \
                                  rtol_emi, rtol_knp, atol_emi, atol_knp, \
                                  threshold_emi, threshold_knp)

    # File for results
    fname = "results/data/two_tags/"

    # Dictionary with membrane models (key is facet tag, value is ode model)
    ode_models = {1: mm_hh, 2:mm_glial}

    # Solve system
    S = Solver(params, ion_list)                    # create solver
    S.setup_domain(mesh, subdomains, surfaces)      # setup meshes
    S.setup_parameters()                            # setup physical parameters
    S.setup_FEM_spaces()                            # setup function spaces and numerical parameters
    S.setup_membrane_model(stim_params, ode_models) # setup membrane model(s)
    S.solve_system_active(Tstop, t, solver_params, filename=fname) # solve
