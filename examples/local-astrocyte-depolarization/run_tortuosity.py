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

from knpemidg.utils import pcws_constant_project
from knpemidg.utils import interface_normal, plus, minus

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

class Solver3D(Solver):
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
        Na_i = minus(self.c_prev_k.split()[1], self.n_g)
        ode_model.set_parameter('Na_i', pcws_constant_project(Na_i, self.Q))

        return

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

    # Resolution factor of mesh
    resolution = 0

    # Time variables (PDEs)
    dt = 0.1                         # global time step (ms)
    Tstop = 10                       # ms

    t = Constant(0.0)                # time constant

    # Time variables (ODEs)
    n_steps_ODE = 25                 # number of ODE steps

    # Physical parameters
    C_M = 1.0                        # capacitance
    temperature = 307e3              # temperature (mK)
    F = 96500e3                      # Faraday's constant (mC/mol)
    R = 8.315e3                      # Gas Constant (mJ/(K*mol))

    D_Na = 1.33e-8                   # diffusion coefficients Na (cm/ms)
    D_K = 1.96e-8                    # diffusion coefficients K (cm/ms)
    D_Cl = 2.03e-8                   # diffusion coefficients Cl (cm/ms)

    psi = F / (R * temperature)      # shorthand
    C_phi = C_M / dt                 # shorthand

    # Initial values potassium
    K_e_init = 3.092970607490389
    K_g_init = 99.3100014897692
    K_n_init = 124.13988964240784

    # Initial values sodium
    Na_e_init = 144.60625137617149
    Na_g_init = 15.775818906083778
    Na_n_init = 12.850454639128186

    # Initial values chloride
    Cl_e_init = 133.62525154406637
    Cl_g_init = 5.203660274163705
    Cl_n_init = 5.0

    # background charge / immobile ions
    rho_e = - (Na_e_init + K_e_init - Cl_e_init)
    rho_g = - (Na_g_init + K_g_init - Cl_g_init)
    rho_n = - (Na_n_init + K_n_init - Cl_n_init)

    rho_sub = {0:Constant(rho_e), 1:Constant(rho_n), 2:Constant(rho_g)}

    # short stimuli regime
    t_syn = 1.2

    # baseline
    #lambda_i = 3.2
    #lambda_e = 1.6
    #fname = "results/data/EMIx-synapse_100_tort_short_baseline/"
    #g_syn = 200

    # M1: increase ECS and ICS lambda
    #lambda_i = 3.2*2
    #lambda_e = 1.6*2
    #fname = "results/data/EMIx-synapse_100_tort_short_2_tort_both_ICS_and_ECS/"
    #g_syn = 65

    # M2: increase ECS and ICS lambda
    lambda_i = 3.2*4
    lambda_e = 1.6*4
    fname = "results/data/EMIx-synapse_100_tort_short_4_tort_both_ICS_and_ECS/"
    g_syn = 26

    phi_M_init_type = 'constant'

    # Set physical parameters
    params = namedtuple('params', ('dt', 'n_steps_ODE', 'F', 'psi', \
                        'C_phi', 'C_M', 'R', 'temperature', \
                        'phi_M_init_type', 'rho_sub'))(dt, n_steps_ODE, F, psi, \
                         C_phi, C_M, R, temperature, phi_M_init_type, rho_sub)

    # diffusion coefficients for each sub-domain
    D_Na_sub = {0:Constant(D_Na/(lambda_e**2)), 1:Constant(D_Na/(lambda_i**2)), 2:Constant(D_Na/(lambda_i**2))}
    D_K_sub  = {0:Constant(D_K/(lambda_e**2)),  1:Constant(D_K/(lambda_i**2)), 2:Constant(D_K/(lambda_i**2))}
    D_Cl_sub = {0:Constant(D_Cl/(lambda_e**2)), 1:Constant(D_Cl/(lambda_i**2)), 2:Constant(D_Cl/(lambda_i**2))}

    # initial concentrations for each sub-domain
    Na_init_sub = {0:Constant(Na_e_init), 1:Constant(Na_n_init), \
                   2:Constant(Na_g_init)}
    K_init_sub  = {0:Constant(K_e_init), 1:Constant(K_n_init), \
                   2:Constant(K_g_init)}
    Cl_init_sub = {0:Constant(Cl_e_init), 1:Constant(Cl_n_init), \
                   2:Constant(Cl_g_init)}
    c_init_sub_type = 'constant'

    # long stimuli regime 20 ms
    #g_syn = 50
    #t_syn = 100.2

    # long stimuli regime 20 ms
    #g_syn = 50
    #t_syn = 10.2

    # new ROI
    xmin = 2700e-7; xmax = 3100e-7
    ymin = 1700e-7; ymax = 2100e-7
    zmin = 1800e-7; zmax = 2200e-7

    f_neuron_K = Expression("g_syn*(xmin <= x[0])*(x[0] <= xmax)* \
                                   (ymin <= x[1])*(x[1] <= ymax)* \
                                   (zmin <= x[2])*(x[2] <= zmax)* \
                                   (0.2 <= t)*(t <= t_syn)",
                                   t=t, g_syn=g_syn, t_syn=t_syn,
                                   xmin=xmin, xmax=xmax,
                                   ymin=ymin, ymax=ymax,
                                   zmin=zmin, zmax=zmax,
                                   degree=4)

    f_neuron_Cl = Constant(0)

    f_neuron_Na = Expression("- g_syn*(xmin <= x[0])*(x[0] <= xmax)* \
                                      (ymin <= x[1])*(x[1] <= ymax)* \
                                      (zmin <= x[2])*(x[2] <= zmax)* \
                                      (0.2 <= t)*(t <= t_syn)",
                                      t=t, g_syn=g_syn, t_syn=t_syn,
                                      xmin=xmin, xmax=xmax,
                                      ymin=ymin, ymax=ymax,
                                      zmin=zmin, zmax=zmax,
                                      degree=4)

    # Create ions (channel conductivity is set below in the membrane model)
    Na = {'c_init_sub':Na_init_sub,
          'c_init_sub_type':c_init_sub_type,
          'bdry': Constant((0, 0)),
          'z':1.0,
          'name':'Na',
          'D_sub':D_Na_sub,
          'f_source':f_neuron_Na}

    K = {'c_init_sub':K_init_sub,
         'c_init_sub_type':c_init_sub_type,
         'bdry': Constant((0, 0)),
         'z':1.0,
         'name':'K',
         'D_sub':D_K_sub,
         'f_source':f_neuron_K}

    Cl = {'c_init_sub':Cl_init_sub,
          'c_init_sub_type':c_init_sub_type,
          'bdry': Constant((0, 0)),
          'z':-1.0,
          'name':'Cl',
          'D_sub':D_Cl_sub,
          'f_source':f_neuron_Cl}

    # Create ion list. NB! The last ion in list will be eliminated, and
    # should be the ion with the smallest diffusion coefficient D_k
    ion_list = [K, Na, Cl]

    # Membrane parameters
    g_syn_bar = 0           # synaptic conductivity (mS/cm**2)

    # Set stimulus ODE
    stimulus = {'stim_amplitude': g_syn_bar}
    stimulus_locator = lambda x: True

    stim_params = namedtuple('membrane_params', ('g_syn_bar', \
                             'stimulus', 'stimulus_locator'))(g_syn_bar, \
                              stimulus, stimulus_locator)

    # Get mesh, subdomains, surfaces paths
    #mesh_path = 'meshes/synapse/size+5000/dx+20_ncells+5/envelopsize+18/'
    # File for results
    #fname = "results/data/EMIx-synapse_5/"

    # Get mesh, subdomains, surfaces paths
    #mesh_path = 'meshes/synapse/size+5000/dx+20_ncells+10/envelopsize+18/'
    # File for results
    #fname = "results/data/EMIx-synapse_10/"

    # Get mesh, subdomains, surfaces paths
    #mesh_path = 'meshes/synapse/size+5000/dx+20_ncells+50/envelopsize+18/'
    # File for results
    #fname = "results/data/EMIx-synapse_50/"

    # Get mesh, subdomains, surfaces paths
    mesh_path = 'meshes/synapse/size+5000/dx+20_ncells+100/envelopsize+18/'

    mesh = Mesh()
    infile = XDMFFile(mesh_path + 'mesh.xdmf')
    infile.read(mesh)
    cdim = mesh.topology().dim()
    infile.close()

    # convert mesh from nm to cm
    mesh.coordinates()[:,:] *= 1e-7

    subdomains = MeshFunction('size_t', mesh, mesh_path + 'subdomains.xml')
    surfaces = MeshFunction('size_t', mesh, mesh_path + 'surfaces.xml')

    print(np.unique(subdomains.array()))
    print(np.unique(surfaces.array()))

    # Set solver parameters EMI (True is direct, and False is iterate)
    direct_emi = False
    rtol_emi = 1E-5
    atol_emi = 1E-40
    threshold_emi = 0.9

    # Set solver parameters KNP (True is direct, and False is iterate)
    direct_knp = False
    rtol_knp = 1E-7
    atol_knp = 1E-40
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

    # Dictionary with membrane models (key is facet tag, value is ode model)
    ode_models = {1: mm_hh, 2: mm_glial, 3: mm_hh}

    # Solve system
    S = Solver3D(params, ion_list, sf=1)            # create solver
    S.setup_domain(mesh, subdomains, surfaces)      # setup meshes
    S.setup_parameters()                            # setup physical parameters
    S.setup_FEM_spaces()                            # setup function spaces and numerical parameters
    S.setup_membrane_model(stim_params, ode_models) # setup membrane model(s)
    S.solve_system_active(Tstop, t, solver_params, 
            filename=fname, save_fields=True, save_solver_stats=True) # solve
