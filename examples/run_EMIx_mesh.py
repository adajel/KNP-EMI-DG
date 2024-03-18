#!/usr/bin/python3

import os
import sys
import time

from dolfin import *
import numpy as np

from collections import namedtuple

from solver import Solver
from membrane import MembraneModel
import mm_hh as ode
import mm_leak as leak

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

class Solver3DRatNeuron(Solver):

    def __init__(self, params, ion_list, degree_emi=1, degree_knp=1, mms=None):
        super().__init__(params, ion_list, degree_emi=1, degree_knp=1, mms=None)

    def solve_system_active(self, Tstop, t, solver_params, membrane_params, filename=None):
        """ Solve KNP-EMI system with active membrane mechanisms (ODEs) """

        # Setup solver and parameters
        self.solver_params = solver_params          # parameters for solvers
        self.direct_emi = solver_params.direct_emi  # choice of emi solver
        self.direct_knp = solver_params.direct_knp  # choice of knp solver
        self.splitting_scheme = True                # splitting scheme
        self.setup_parameters()                     # setup physical parameters
        self.setup_FEM_spaces()                     # setup function spaces and numerical parameters

        return

if __name__ == "__main__":

    # Resolution factor of mesh
    resolution = 0

    # Time variables (PDEs)
    dt = 1.0e-4                      # global time step (s)
    Tstop = 1.0                      # global end time (s)
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

    # Membrane parameters
    g_Na_leak = Constant(1.0)        # Na leak conductivity (S/m**2)
    g_K_leak = Constant(4.0)         # K leak conductivity (S/m**2)
    g_Cl_leak = Constant(0.0)        # K leak conductivity (S/m**2)
    g_Na_bar = 1200                  # Na max conductivity (S/m**2)
    g_K_bar = 360                    # K max conductivity (S/m**2)
    g_syn_bar = 40                   # synaptic conductivity (S/m**2)

    # Set parameters
    params = namedtuple('params', ('dt', 'n_steps_ODE', 'F', 'psi', \
            'phi_M_init', 'C_phi', 'C_M', 'R', 'temperature'))(dt, \
            n_steps_ODE, F, psi, phi_M_init, C_phi, C_M, R, temperature)

    # Create ions (channel conductivity is set below for each model)
    Na = {'D1':D_Na, 'D2':D_Na,
          'c1_init':Na_i_init, 'c2_init':Na_e_init, 'bdry': Constant((0, 0)),
          'z':1.0, 'name':'Na'}
    K = {'D1':D_K, 'D2':D_K,
         'c1_init':K_i_init, 'c2_init':K_e_init, 'bdry': Constant((0, 0)),
         'z':1.0, 'name':'K'}
    Cl = {'D1':D_Cl, 'D2':D_Cl,
          'c1_init':Cl_i_init, 'c2_init':Cl_e_init, 'bdry': Constant((0, 0)),
          'z':-1.0, 'name':'Cl'}

    # Create ion list. NB! The last ion in list will be eliminated, and should 
    # be the ion with the smallest diffusion coefficient
    ion_list = [K, Cl, Na]

    # Membrane parameters
    g_Na_bar = 1200                  # Na max conductivity (S/m**2)
    g_K_bar = 360                    # K max conductivity (S/m**2)
    g_syn_bar = 200                  # synaptic conductivity (S/m**2)
    g_Na_leak = Constant(2.0*0.5)    # Na leak conductivity (S/m**2)
    g_K_leak = Constant(8.0*0.5)     # K leak conductivity (S/m**2)
    g_Cl_leak = Constant(0.0)        # K leak conductivity (S/m**2)

    # Set membrane parameters
    membrane_params = namedtuple('membrane_params', ('g_Na_bar',
                                 'g_K_bar', 'g_Na_leak', 'g_K_leak', 'g_Cl_leak', \
                                 'g_syn_bar'))(g_Na_bar, g_K_bar, g_Na_leak, \
                                  g_K_leak, g_Cl_leak, g_syn_bar)

    # Get mesh, subdomains, surfaces paths
    #mesh_path = 'meshes/emix_meshes/volume_ncells_5_size_5000/'
    mesh_path = 'meshes/emix_meshes/volume_ncells_50_size_5000/'
    #mesh_path = 'meshes/emix_meshes/volume_ncells_200_size_5000/'

    mesh = Mesh()
    infile = XDMFFile(mesh_path + 'mesh.xdmf')
    infile.read(mesh)
    cdim = mesh.topology().dim()
    subdomains = MeshFunction("size_t", mesh, cdim)
    infile.read(subdomains, "label")
    infile.close()

    # Remark subdomains
    for cell in cells(mesh):
        if subdomains[cell] == 1:
            subdomains[cell] = 2
        else:
            subdomains[cell] = 1

    # get all local labels
    File("meshes/emix_meshes/subdomains.pvd") << subdomains

    infile = XDMFFile(mesh_path + 'facets.xdmf')
    surfaces = MeshFunction("size_t", mesh, cdim - 1)
    infile.read(surfaces, "boundaries")
    infile.close()

    print(np.unique(surfaces.array()))

    unique, counts = np.unique(surfaces.array(), return_counts=True)

    print(dict(zip(unique, counts)))

    int_fac_mem = 0
    int_fac_int = 1
    # Remark facets
    for facet in facets(mesh):
        if surfaces[facet] > 0:
            surfaces[facet] = 1
            int_fac_mem +=1
        elif surfaces[facet] > 1000:
            surfaces[facet] = 5
        elif surfaces[facet] == 0:
            int_fac_int +=1

    print(int_fac_int)
    print(int_fac_mem)
    print(int_fac_mem/len(surfaces)*100)

    File("meshes/emix_meshes/surfaces.pvd") << surfaces

    # Set solver parameters (True is direct, and False is iterate)
    direct_emi = False
    direct_knp = False

    solver_params = namedtuple('solver_params', ('direct_emi',
        'direct_knp', 'resolution'))(direct_emi, direct_knp, resolution)

    # File for saving results
    fname = "results/data/emix_mesh/"

    # Solve system
    S_1a = Solver3DRatNeuron(params, ion_list, degree_emi=1, degree_knp=1) # create solver
    S_1a.setup_domain(mesh, subdomains, surfaces)                          # setup domains

    S_1a.solve_system_active(Tstop, t, solver_params, membrane_params, \
            filename=fname)                                                # solve
