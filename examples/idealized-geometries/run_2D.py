#!/usr/bin/python3

from dolfin import *

import os
import sys
import time

import math as ma
import numpy as np

from knpemidg import Solver
import mm_hh as mm_hh

from knpemidg.utils import pcws_constant_project
from knpemidg.utils import interface_normal, plus, minus

# define colors for printing
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

class Solver2D(Solver):
    """ sub-class for solving 2D problem """

    def __init__(self, params, ion_list, degree_emi=1, degree_knp=1, mms=None, sf=1):
        Solver.__init__(self, params, ion_list, degree_emi=1, degree_knp=1, mms=None, sf=1)

        return

    def update_ode(self, ode_model):
        """ Update parameters in ODE solver (based on previous PDEs step)
            specific to membrane model """

        # set extracellular trace of K concentration at membrane
        K_e = plus(self.c_prev_k.split()[0], self.n_g)
        ode_model.set_parameter('K_e', pcws_constant_project(K_e, self.Q))

        # set intracellular trace of Na concentration at membrane
        Na_i = minus(self.ion_list[-1]['c'], self.n_g)
        ode_model.set_parameter('Na_i', pcws_constant_project(Na_i, self.Q))

        return

if __name__=='__main__':

    from collections import namedtuple

    # Resolution factor of mesh
    #for resolution in [2, 3, 4, 5]:
    for resolution in [2]:

        # Time variables (PDEs)
        dt = 1.0e-4                      # global time step (s)
        #Tstop = 3.0e-1                   # global end time (s)
        Tstop = 2.0e-2                   # global end time (s)
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
        Na_i_init = 12.838513108648856   # Intracellular Na concentration
        Na_e_init = 100.71925900027354   # extracellular Na concentration
        K_i_init = 124.15397583491901    # intracellular K concentration
        K_e_init = 3.3236967382705265    # extracellular K concentration
        Cl_e_init = Na_e_init + K_e_init # extracellular CL concentration
        Cl_i_init = Na_i_init + K_i_init # intracellular CL concentration
        phi_M_init = Constant(-0.07438609374462003)   # membrane potential (V)
        phi_M_init_type = 'constant'

        # set background charge (no background charge in this scenario)
        rho_sub = {0:Constant(0), 1:Constant(0), 2:Constant(0)}

        # Set parameters
        params = namedtuple('params', ('dt', 'n_steps_ODE', 'F', 'psi',
            'phi_M_init', 'C_phi', 'C_M', 'R', 'temperature',
            'phi_M_init_type', 'rho_sub'))(dt, n_steps_ODE, \
             F, psi, phi_M_init, C_phi, C_M, R, temperature, \
             phi_M_init_type, rho_sub)

        # diffusion coefficients for each sub-domain
        D_Na_sub = {1:D_Na, 0:D_Na}
        D_K_sub = {1:D_K, 0:D_K}
        D_Cl_sub = {1:D_Cl, 0:D_Cl}

        # initial concentrations for each sub-domain
        Na_init_sub = {1:Constant(Na_i_init), 0:Constant(Na_e_init)}
        K_init_sub = {1:Constant(K_i_init), 0:Constant(K_e_init)}
        Cl_init_sub = {1:Constant(Cl_i_init), 0:Constant(Cl_e_init)}
        c_init_sub_type = 'constant'

        # set source terms to be zero for all ion species
        f_source_Na = Constant(0)
        f_source_K = Constant(0)
        f_source_Cl = Constant(0)

        # Create ions (channel conductivity is set below for each model)
        Na = {'c_init_sub':Na_init_sub,
              'c_init_sub_type':c_init_sub_type,
              'bdry': Constant((0, 0)),
              'z':1.0,
              'name':'Na',
              'D_sub':D_Na_sub,
              'f_source':f_source_Na}

        K = {'c_init_sub':K_init_sub,
             'c_init_sub_type':c_init_sub_type,
             'bdry': Constant((0, 0)),
             'z':1.0,
             'name':'K',
             'D_sub':D_K_sub,
             'f_source':f_source_K}

        Cl = {'c_init_sub':Cl_init_sub,
              'c_init_sub_type':c_init_sub_type,
              'bdry': Constant((0, 0)),
              'z':-1.0,
              'name':'Cl',
              'D_sub':D_Cl_sub,
              'f_source':f_source_Cl}

        # Create ion list. NB! The last ion in list will be eliminated
        ion_list = [K, Cl, Na]

        # Membrane parameters
        g_syn_bar = 10                   # synaptic conductivity (S/m**2)

        # set stimulus ODE
        stimulus = {'stim_amplitude': g_syn_bar}
        stimulus_locator = lambda x: (x[0] < 20e-6)

        # Set membrane parameters
        stim_params = namedtuple('membrane_params', ('g_syn_bar', \
                                 'stimulus', 'stimulus_locator'))(g_syn_bar, \
                                  stimulus, stimulus_locator)

        # Get mesh, subdomains, surfaces paths
        here = os.path.abspath(os.path.dirname(__file__))
        mesh_prefix = os.path.join(here, 'meshes/2D/')
        mesh_path = mesh_prefix + 'mesh_' + str(resolution) + '.xml'
        subdomains_path = mesh_prefix + 'subdomains_' + str(resolution) + '.xml'
        surfaces_path = mesh_prefix + 'surfaces_' + str(resolution) + '.xml'

        # generate mesh if it does not exist
        if not os.path.isfile(mesh_path):
            from make_mesh_2D import main
            main(["-r", str(resolution), "-d", mesh_prefix])

        mesh = Mesh(mesh_path)
        subdomains = MeshFunction('size_t', mesh, subdomains_path)
        surfaces = MeshFunction('size_t', mesh, surfaces_path)

        # Set solver parameters EMI (True is direct, and False is iterate) 
        direct_emi = False
        rtol_emi = 1E-5
        atol_emi = 1E-40
        threshold_emi = None

        # Set solver parameters KNP (True is direct, and False is iterate) 
        direct_knp = False
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
        fname = 'results/data/2D/'

        # Dictionary with membrane models (key is facet tag, value is ode model)
        ode_models = {1: mm_hh}

        # Solve system
        S = Solver2D(params, ion_list)                    # create solver
        S.setup_domain(mesh, subdomains, surfaces)      # setup meshes
        S.setup_parameters()                            # setup physical parameters
        S.setup_FEM_spaces()                            # setup function spaces and numerical parameters
        S.setup_membrane_model(stim_params, ode_models) # setup membrane model(s)
        S.solve_system_active(Tstop, t, solver_params, \
                filename=fname, save_fields=True, save_solver_stats=True) # solve
