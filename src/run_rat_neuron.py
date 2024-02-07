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

        # Get membrane parameters
        self.membrane_params = membrane_params      # parameters for membrane model(s)
        g_Na_bar = membrane_params.g_Na_bar         # Na max conductivity (S/m**2)
        g_K_bar = membrane_params.g_K_bar           # K max conductivity (S/m**2)
        g_Na_leak = membrane_params.g_Na_leak       # Na leak conductivity (S/m**2)
        g_K_leak = membrane_params.g_K_leak         # K leak conductivity (S/m**2)
        g_Cl_leak = membrane_params.g_Cl_leak       # Cl leak conductivity (S/m**2)
        g_syn_bar = membrane_params.g_syn_bar       # synaptic conductivity (S/m**2)

        self.filename = filename

        # Create membrane model
        membrane_leak = MembraneModel(ode, facet_f=self.surfaces, tag=1, V=self.Q)
        membrane_ode = MembraneModel(ode, facet_f=self.surfaces, tag=2, V=self.Q)

        # set stimulus ODE
        stimulus = {'stim_amplitude': g_syn_bar}
        stimulus_locator = lambda x: (x[1] < -80e-6) or (x[0] < -125e-6) or (x[0] > 140e-6)

        g_syn = Expression('g_syn_bar * exp(-fmod(t, 0.02)/0.002) * \
                           ((x[1] < -80e-6) + (x[0] < -125e-6) + (x[0] > \
                           140e-6))', t=t, g_syn_bar=g_syn_bar, degree=4)

        # Initialize gating variables ODEs
        self.n_HH = Function(self.Q)
        self.m_HH = Function(self.Q)
        self.h_HH = Function(self.Q)

        # Define conductivities for membrane models dendrite (tag 1)
        g_K_1 = g_K_leak
        g_Cl_1 = g_Cl_leak
        g_Na_1 = g_Na_leak + g_syn

        # Define conductivities for membrane model soma and axon (tag 2)
        g_K_2 = g_K_leak + Constant(g_K_bar)*self.n_HH**4
        g_Cl_2 = g_Cl_leak
        g_Na_2 = g_Na_leak + Constant(g_Na_bar)*self.m_HH**3*self.h_HH
        #g_Na_2 = g_Na_leak + g_syn + Constant(g_Na_bar)*self.m_HH**3*self.h_HH

        # Set membrane models and associated tags
        mem_models_K  = [{'tag':1, 'g_k':g_K_1}, {'tag':2, 'g_k':g_K_2}]
        mem_models_Cl = [{'tag':1, 'g_k':g_Cl_1}, {'tag':2, 'g_k':g_Cl_2}]
        mem_models_Na = [{'tag':1, 'g_k':g_Na_1}, {'tag':2, 'g_k':g_Na_2}]

        # Set tags (NB! Must match with tags in membrane models)
        self.mem_tags = [1, 2]

        # Assign membrane models
        self.ion_list[0]['mem_models'] = mem_models_K
        self.ion_list[1]['mem_models'] = mem_models_Cl
        self.ion_list[2]['mem_models'] = mem_models_Na

        # Set ODE parameters (to ensure same values are used)
        membrane_ode.set_parameter_values({'g_K_bar': lambda x: g_K_bar})
        membrane_ode.set_parameter_values({'g_Na_bar': lambda x: g_Na_bar})
        membrane_ode.set_parameter_values({'Cm': lambda x: self.params.C_M})
        membrane_ode.set_parameter_values({'g_leak_K': lambda x: g_K_leak})
        membrane_ode.set_parameter_values({'g_leak_Na': lambda x: g_Na_leak})

        # Set passive parameters (to ensure same values are used)
        membrane_leak.set_parameter_values({'Cm': lambda x: self.params.C_M})
        membrane_leak.set_parameter_values({'g_leak_K': lambda x: g_K_leak})
        membrane_leak.set_parameter_values({'g_leak_Na': lambda x: g_Na_leak})

        # Setup variational formulations
        self.setup_varform_emi()
        self.setup_varform_knp()

        # Setup solvers
        self.setup_solver_emi()
        self.setup_solver_knp()

        # Calculate ODE time step (s)
        dt_ode = float(self.dt/self.params.n_steps_ODE)

        # Initialize save results
        if filename is not None:
            # file for solutions to equations
            self.initialize_h5_savefile(filename + 'results.h5')
            # file for CPU timings, number of iterations etc.
            self.initialize_solver_savefile(filename + 'solver/')

        # Solve system (PDEs and ODEs)
        for k in range(int(round(Tstop/float(self.dt)))):
            # Start timer (ODE solve)
            ts = time.perf_counter()

            # -------------------------------------------------- #
            # Solve ODEs active part of membrane
            # -------------------------------------------------- #

            # Update initial values and parameters in ODE solver
            membrane_ode.set_membrane_potential(self.phi_M_prev_PDE)
            membrane_ode.set_parameter('E_K', self.ion_list[0]['E'])
            membrane_ode.set_parameter('E_Na', self.ion_list[-1]['E'])

            # Solve ODEs
            membrane_ode.step_lsoda(dt=dt_ode*self.params.n_steps_ODE, \
                    stimulus=None, stimulus_locator=stimulus_locator)

            # Update PDE functions based on ODE output
            membrane_ode.get_membrane_potential(self.phi_M_prev_PDE)
            membrane_ode.get_state('n', self.n_HH)
            membrane_ode.get_state('m', self.m_HH)
            membrane_ode.get_state('h', self.h_HH)

            # -------------------------------------------------- #
            # Solve ODEs passive part of membrane
            # -------------------------------------------------- #

            # Update initial values and parameters in passive model
            membrane_leak.set_membrane_potential(self.phi_M_prev_PDE)
            membrane_leak.set_parameter('E_K', self.ion_list[0]['E'])
            membrane_leak.set_parameter('E_Na', self.ion_list[-1]['E'])

            # Solve ODEs
            membrane_leak.step_lsoda(dt=dt_ode*self.params.n_steps_ODE, \
                    stimulus=stimulus, stimulus_locator=stimulus_locator)

            # Update PDE functions based on ODE output
            membrane_leak.get_membrane_potential(self.phi_M_prev_PDE)

            # End timer (ODE solve)
            te = time.perf_counter()
            res = te - ts
            print(f"{bcolors.OKGREEN} CPU Execution time ODE solve: {res:.4f} seconds {bcolors.ENDC}")

            # Solve PDEs
            self.solve_for_time_step(k, t)
            #self.solve_for_time_step_picard(k, t)

            # Save results
            if (k % self.sf) == 0 and filename is not None:
                self.save_h5()      # fields
                self.save_solver(k) # solver statistics

        # Close files
        if filename is not None:
            self.close_h5()
            self.close_save_solver()

        # Combine solution for the potential and concentrations
        uh = split(self.c) + (self.phi,)

        return uh

if __name__ == "__main__":

    # Resolution factor of mesh
    resolution = 0

    # Time variables (PDEs)
    dt = 1.0e-4                      # global time step (s)
    #Tstop = 1.0                     # global end time (s)
    Tstop = 1.0e-2                   # global end time (s)
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
            cell_f[cell] = 2
        elif cell_f[cell] == 2:
            cell_f[cell] = 1
        else:
            print("no cell found")

    # Print for debug
    nzero_cells = np.sum(cell_f.array() == 1)
    none_cells = np.sum(cell_f.array() == 2)

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

            #if (-80 < f.midpoint().y() < -70):
                #print(f.midpoint().x(), f.midpoint().y(), f.midpoint().z())

            facet_f[f.index()] = 1
            #if f.midpoint().y() >= -6.0:
            if f.midpoint().y() >= -5.0:
                facet_f[f.index()] = 2

    # File for testing marking (by eye in paraview)
    #File('test_subdomains.pvd') << cell_f
    #File('test_interfaces.pvd') << facet_f

    # Convert mesh to unit meter (m)
    mesh.coordinates()[:, :] *= 1e-6

    subdomains = cell_f
    surfaces = facet_f

    File("meshes/rat_neuron/surfaces.pvd") << surfaces
    File("meshes/rat_neuron/subdomains.pvd") << subdomains

    # Set solver parameters (True is direct, and False is iterate)
    direct_emi = False
    direct_knp = False

    solver_params = namedtuple('solver_params', ('direct_emi',
        'direct_knp', 'resolution'))(direct_emi, direct_knp, resolution)

    # File for saving results
    fname = "results/data/rat_neuron/"

    # Solve system
    S_1a = Solver3DRatNeuron(params, ion_list, degree_emi=1, degree_knp=1) # create solver
    S_1a.setup_domain(mesh, subdomains, surfaces)                          # setup domains
    S_1a.solve_system_active(Tstop, t, solver_params, membrane_params, \
            filename=fname)                                                # solve
