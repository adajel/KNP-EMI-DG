from dolfin import *
import numpy as np
import sys
import os
import time
from petsc4py import PETSc

from knpemidg.utils import pcws_constant_project
from knpemidg.utils import interface_normal, plus, minus
from knpemidg.utils import CellCenterDistance
from knpemidg.membrane import MembraneModel

# define jump across the membrane (interface gamma)
JUMP = lambda f, n: minus(f, n) - plus(f, n)

parameters['ghost_mode'] = 'shared_vertex'

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

# We here approximate the following system:
#     d(c_k)/dt + div(J_k) = 0, (emi)
#   - F sum_k z^k div(J_k) = 0, (knp)
#   where
#   J_k(c_k, phi) = - D grad(c_k) - z_k D_k psi c_k grad(phi)
#
#   We solve the system iteratively, by decoupling the first and second
#   equation, yielding the following system: Given c_k_ and phi_M_
#   iterate over the two following steps:
#
#       step I:  (emi) find phi by solving (2), with J^k(c_k_, phi)
#       step II: (knp) find c_k by solving (1) with J^k(c_k, phi), where phi
#                is the solution from step I
#      (step III: solve ODEs at interface, and update membrane potential)
#
# Membrane potential is defined as phi_i - phi_e, since we have
# marked cell in ECS with 0 and cells in ICS with 1 we have an
# interface normal pointing inwards (from lower to higher)
#    ____________________
#   |                    |
#   |      ________      |
#   |     |        |     |
#   | ECS |   ICS  |     |
#   |  0  |->  1   |     |
#   | (+) |   (-)  |     |
#   |     |________|     |
#   |                    |
#   |____________________|
#
# Normal will always point from lower to higher (e.g. from 0 -> 1)
# NB! The code assumes that all interior facets are tagged with 0.

class SolverEMI:
    def __init__(self, params, ion_list, degree_emi=1, degree_knp=1, mms=None, sf=1):
        """
        Initialize solver
        """

        self.ion_list = ion_list            # list of ions species
        self.N_ions = len(ion_list[:-1])    # number of ions
        self.degree_emi = degree_emi        # polynomial degree EMI subproblem
        self.degree_knp = degree_knp        # polynomial degree KNP subproblem
        self.mms = mms                      # boolean for mms test
        self.params = params                # physical parameters
        self.sf = sf                        # frequency for saving results

        # timers
        self.ode_solve_timer = 0
        self.emi_solve_timer = 0
        self.emi_ass_timer = 0

        return

    def setup_domain(self, mesh, subdomains, surfaces):
        """
        Setup mesh and associated parameters
        """

        # set mesh, subdomains, and surfaces
        self.mesh = mesh
        self.subdomains = subdomains
        self.surfaces = surfaces

        # define measures
        self.dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
        self.ds = Measure('ds', domain=mesh, subdomain_data=surfaces)
        self.dS = Measure('dS', domain=mesh, subdomain_data=surfaces)

        # facet area and normal
        self.n = FacetNormal(mesh)
        self.hA_emi = CellDiameter(self.mesh)

        # interface normal
        self.n_g = interface_normal(subdomains, mesh)

        # DG penalty parameters
        self.gdim = self.mesh.geometry().dim()
        self.tau_emi = Constant(20*self.gdim*self.degree_emi)

        # DG elements for ion concentrations and the potential
        self.PK_emi = FiniteElement('DG', mesh.ufl_cell(), self.degree_emi)
        self.PK_knp = self.PK_emi

        # For the MMS problem, we need unique tags for each of the interface walls
        if self.mms is not None:
            self.lm_tags = [1, 2, 3, 4]

        return


    def setup_parameters(self):
        """
        Setup physical parameters
        """

        # setup physical parameters
        params = self.params

        # get physical parameters
        self.C_phi = Constant(params.C_phi)             # coupling coefficient
        self.C_M = Constant(params.C_M)                 # capacitance
        self.dt = Constant(params.dt)                   # time step
        self.F = Constant(params.F)                     # Faraday constant
        self.R = Constant(self.params.R)                # Gas constant
        self.temperature = Constant(params.temperature) # temperature
        self.phi_M_init = params.phi_M_init             # initial membrane potential
        self.psi = self.F/(self.R*self.temperature)     # shorthand

        for idx, ion in enumerate(self.ion_list):
            # define global diffusion coefficients (for each ion)
            D = self.make_global(ion['D_sub'])
            ion['D'] = D

            # define global coupling coefficient (for MMS case, for each ion)
            if self.mms is not None:
                C = self.make_global(ion['C_sub'])
                ion['C'] = C

        return


    def setup_FEM_spaces(self):
        """
        Setup function spaces and functions
        """

        # create function space for potential (phi)
        self.V_emi = FunctionSpace(self.mesh, self.PK_emi)
        # function for solution potential
        self.phi = Function(self.V_emi)

        # set up finite element space for concentrations (c)
        ME = MixedElement([self.PK_knp]*self.N_ions)
        self.V_knp = FunctionSpace(self.mesh, ME)

        # function for previous solution concentrations in time stepping
        self.c_prev_n = Function(self.V_knp)
        # function for previous solution concentrations in Picard iteration
        self.c_prev_k = Function(self.V_knp)

        ion_list = self.ion_list

        # set initial conditions concentrations
        for idx, ion in enumerate(self.ion_list):

            # interpolate initial conditions to global function and assign
            c_init = self.make_global(ion['c_init_sub'])

            if idx == len(ion_list) - 1:
                # set initial concentrations for eliminated ion
                ion_list[-1]['c'] = interpolate(c_init, self.V_knp.sub(self.N_ions - 1).collapse())
            else:
                assign(self.c_prev_n.sub(idx), interpolate(c_init, self.V_knp.sub(idx).collapse()))
                assign(self.c_prev_k.sub(idx), interpolate(c_init, self.V_knp.sub(idx).collapse()))

        # define function space of piecewise constants on interface gamma for solution to ODEs
        self.Q = FunctionSpace(self.mesh, 'Discontinuous Lagrange Trace', 0)

        # set initial membrane potential
        self.phi_M_prev_PDE = pcws_constant_project(self.phi_M_init, self.Q)

        return


    def setup_membrane_model(self, stim_params, odes):
        """
        Initiate membrane model(s) containing membrane mechanisms (passive
        dynamics / ODEs) and src terms for PDE system
        """

        # set stimulus parameters
        self.stimulus = stim_params.stimulus                 # stimulus
        self.stimulus_locator = stim_params.stimulus_locator # locator for stimulus

        # list of membrane models
        self.mem_models = []

        # initialize and append ode models to list
        for tag, ode in odes.items():
            # Initialize ODE model
            ode_model = MembraneModel(ode, facet_f=self.surfaces, tag=tag, V=self.Q)

            # Set ODE capacitance (to ensure same values are used)
            ode_model.set_parameter_values({'Cm': lambda x: self.params.C_M})

            # dictionary for ion specific currents (i.e src terms PDEs)
            I_ch_k = {}
            # Initialize src terms for PDEs
            for ion in self.ion_list:
                # function for src term pde
                I_ch_k_ = Function(self.Q)
                # set src terms pde
                ode_model.get_parameter("I_ch_" + ion['name'], I_ch_k_)
                # set function in dictionary
                I_ch_k[ion['name']] = I_ch_k_

            # define membrane model (with ode model and src term for PDEs)
            mem_model = {'ode': ode_model, 'I_ch_k': I_ch_k}

            # append to list of membrane models
            self.mem_models.append(mem_model)

            return


    def setup_varform_emi(self):
        """ setup variational form for the emi system """

        dx = self.dx; ds = self.ds; dS = self.dS     # measures
        hA = self.hA_emi; n = self.n; n_g = self.n_g # facet area and normal
        tau_emi = self.tau_emi                       # penalty parameter
        ion_list = self.ion_list                     # ion_list
        C_phi = self.C_phi; F = self.F               # physical parameters
        psi = self.psi; C_M = self.C_M               # physical parameters
        R = self.R; temperature = self.temperature   # physical parameters

        # test and trial functions for potential
        u_phi = TrialFunction(self.V_emi)
        v_phi = TestFunction(self.V_emi)

        # initialize
        self.kappa = 0                       # kappa
        a = 0; L = 0                         # rhs and lhs forms
        self.alpha_sum = 0                   # sum of fractions intracellular

        for idx, ion in enumerate(ion_list):

            if idx == len(ion_list) - 1:
                # get eliminated concentrations from previous global step
                c_k_ = ion_list[-1]['c']
            else:
                # get concentrations from previous global step
                c_k_ = split(self.c_prev_k)[idx]

            # calculate and set Nernst potential for current ion (+ is ECS, - is ICS)
            E = R * temperature / (F * ion['z']) * ln(plus(c_k_, n_g) / minus(c_k_, n_g))
            ion['E'] = pcws_constant_project(E, self.Q)

            # update alpha
            self.alpha_sum += ion['D'] * ion['z'] * ion['z'] * c_k_

            # global kappa
            self.kappa += F * ion['z'] * ion['z'] * ion['D'] * psi * c_k_

            # Add terms rhs (diffusive terms)
            L += - F * ion['z'] * inner((ion['D'])*grad(c_k_), grad(v_phi)) * dx \
                 + F * ion['z'] * inner(dot(avg((ion['D'])*grad(c_k_)), n('+')), jump(v_phi)) * dS(0) \

        # if not MMS, calculate total ionic current
        if self.mms is None:
            # sum of ion specific channel currents for each membrane tag
            self.I_ch = [0]*len(self.mem_models)

            # loop though membrane models to set total ionic current
            for jdx, mm in enumerate(self.mem_models):
                # loop through ion species
                for key, value in mm['I_ch_k'].items():
                    # update total channel current for each tag
                    self.I_ch[jdx] += mm['I_ch_k'][key]

        # equation potential (drift terms)
        a += inner(self.kappa*grad(u_phi), grad(v_phi)) * dx \
           - inner(dot(avg(self.kappa*grad(u_phi)), n('+')), jump(v_phi)) * dS(0) \
           - inner(dot(avg(self.kappa*grad(v_phi)), n('+')), jump(u_phi)) * dS(0) \
           + tau_emi/avg(hA) * inner(avg(self.kappa)*jump(u_phi), jump(v_phi)) * dS(0)

        if self.mms is None:
            # coupling condition at interface
            if self.splitting_scheme:
                # robin condition with splitting
                g_robin_emi = [self.phi_M_prev_PDE]*len(self.mem_models)
            else:
                # original robin condition (without splitting)
                g_robin_emi = [self.phi_M_prev_PDE - (1 / C_phi) * I for I in self.I_ch]

            for jdx, mm in enumerate(self.mem_models):
                # get tag
                tag = mm['ode'].tag

                # add robin condition at interface
                L += C_phi * inner(avg(g_robin_emi[jdx]), JUMP(v_phi, n_g)) * dS(tag)
                # add coupling term at interface
                a += C_phi * inner(jump(u_phi), jump(v_phi))*dS(tag)

        # add terms for manufactured solutions test
        if self.mms is not None:
            lm_tags = self.lm_tags
            g_robin_emi = self.mms.rhs['bdry']['u_phi']
            fphi1 = self.mms.rhs['volume_phi_1']
            fphi2 = self.mms.rhs['volume_phi_2']
            g_flux_cont = self.mms.rhs['bdry']['stress']
            phi1e = self.mms.solution['phi_1']
            phi2e = self.mms.solution['phi_2']

            # add robin condition at interface
            L += sum(C_phi * inner(g_robin_emi[tag], JUMP(v_phi, self.n_g)) * dS(tag) for tag in lm_tags)

            # add coupling term at interface
            a += sum(C_phi * inner(jump(u_phi), jump(v_phi))*dS(tag) for tag in lm_tags)

            # MMS specific: add source terms
            L += inner(fphi1, v_phi)*dx(1) \
               + inner(fphi2, v_phi)*dx(0) \

            # MMS specific: we don't have normal cont. of I_M across interface
            L += sum(inner(g_flux_cont[tag], plus(v_phi, n_g)) * dS(tag) for tag in lm_tags)

            # Neumann
            for idx, ion in enumerate(ion_list):
                # MMS specific: add neumann boundary terms (not zero in MMS case)
                L += - F * ion['z'] * dot(ion['bdry'], n) * v_phi * ds

        # setup preconditioner
        up, vp = TrialFunction(self.V_emi), TestFunction(self.V_emi)

        # scale mass matrix to get condition number independent from domain length
        mesh = self.mesh
        gdim = mesh.geometry().dim()

        for axis in range(gdim):
            x_min = mesh.coordinates().min(axis=0)
            x_max = mesh.coordinates().max(axis=0)

            x_min = np.array([MPI.min(mesh.mpi_comm(), xi) for xi in x_min])
            x_max = np.array([MPI.max(mesh.mpi_comm(), xi) for xi in x_max])

        # scaled mess matrix
        Lp = Constant(max(x_max - x_min))
        # self.B_emi is singular so we add (scaled) mass matrix
        mass = self.kappa*(1/Lp**2)*inner(up, vp)*dx

        B = a + mass

        # set forms lhs (A) and rhs (L)
        self.a_emi = a
        self.L_emi = L
        self.B_emi = B
        self.u_phi = u_phi

        return


    def setup_solver_emi(self):
        """ setup KSP solver for the emi sub-problem """

        # create solver
        ksp = PETSc.KSP().create()

        if self.direct_emi:
            # set options for direct emi solver
            opts = PETSc.Options("EMI_DIR") # get options
            opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix
            opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix

            ksp.setOptionsPrefix("EMI_DIR")
            ksp.setFromOptions()            # update ksp with options set above
            pc = ksp.getPC()                # get pc
            pc.setType("lu")                # set solver to LU
            pc.setFactorSolverType("mumps") # set LU solver to use mumps
        else:
            # set options for iterative emi solver
            opts = PETSc.Options('EMI_ITER')
            opts.setValue('ksp_type', 'cg')
            opts.setValue('ksp_monitor_true_residual', None)
            opts.setValue('ksp_error_if_not_converged', 1)
            opts.setValue('ksp_max_it', 1000)
            opts.setValue('ksp_converged_reason', None)
            opts.setValue('ksp_initial_guess_nonzero', 1)
            opts.setValue('ksp_view', None)
            opts.setValue('pc_type', 'hypre')

            # set tolerances
            opts.setValue('ksp_rtol', self.rtol_emi)
            opts.setValue('ksp_atol', self.atol_emi)
            if self.threshold_emi is not None:
                opts.setValue('pc_hypre_boomeramg_strong_threshold', self.threshold_emi)

            ksp.setOptionsPrefix('EMI_ITER')
            ksp.setConvergenceHistory()
            ksp.setFromOptions()

        # set emi solver
        self.ksp_emi = ksp

        ts = time.perf_counter()

        # assemble system
        AA_emi, bb_emi = map(assemble, (self.a_emi, self.L_emi))
        BB_emi = assemble(self.B_emi)

        te = time.perf_counter()
        self.emi_ass_timer += te - ts

        self.AA_emi = as_backend_type(AA_emi)
        self.BB_emi = as_backend_type(BB_emi)
        self.bb_emi = as_backend_type(bb_emi)

        self.x_emi, _ = self.AA_emi.mat().createVecs()

        # get Null space of A
        z = interpolate(Constant(1), self.V_emi).vector()
        self.Z_ = PETSc.NullSpace().create([as_backend_type(z).vec()])

        return

    def solve_emi(self):
        """ solve emi system using either a direct or iterative solver """

        # reassemble operators
        ts = time.perf_counter()                     # timer start

        # reassemble matrices and vector
        assemble(self.a_emi, self.AA_emi)
        assemble(self.L_emi, self.bb_emi)
        assemble(self.B_emi, self.BB_emi)

        # convert matrices and vector
        AA_emi = self.AA_emi.mat()
        BB_emi = self.BB_emi.mat()
        bb_emi = self.bb_emi.vec()

        # set Null space of A
        AA_emi.setNearNullSpace(self.Z_)

        if self.direct_emi:
            self.Z_.remove(bb_emi)

        te = time.perf_counter()                     # timer end

        # print and save CPU time assembly to file
        res = te - ts
        print(f"{bcolors.OKGREEN} CPU Execution time PDE assemble emi: {res:.4f} seconds {bcolors.ENDC}")
        self.emi_ass_timer += res

        if self.filename is not None:
            self.file_emi_assem.write("ass_time: %.4f \n" % (res))

        if self.direct_emi:
            self.ksp_emi.setOperators(AA_emi, AA_emi)
        else:
            self.ksp_emi.setOperators(AA_emi, BB_emi)

        # solve emi system
        ts = time.perf_counter()               # timer start
        self.ksp_emi.solve(bb_emi, self.x_emi) # solve
        te = time.perf_counter()               # timer end

        # print and save CPU time solve to file
        res = te - ts
        print(f"{bcolors.OKGREEN} CPU Execution time PDE solve emi: {res:.4f} seconds {bcolors.ENDC}")
        self.emi_solve_timer += res

        if not self.direct_emi:
            # print and write number of iterations
            niter = self.ksp_emi.getIterationNumber()

            if self.filename is not None:
                self.file_emi_niter.write("niter: %d \n" % niter)

        if self.filename is not None:
            self.file_emi_solve.write("solve_time: %.4f \n" % (res))

        self.phi.vector().vec().array_w[:] = self.x_emi.array_r[:]
        # make assign above work in parallel
        self.phi.vector().vec().ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        return


    def solve_for_time_step(self, k, t):
        """ solve system for one global time step dt"""

        print("------------------------------------------------")
        print(f"{bcolors.WARNING} t = {float(t)} {bcolors.ENDC}")
        print(f"{bcolors.WARNING} k = {k} {bcolors.ENDC}")
        print("------------------------------------------------")

        # Step I: solve emi equations with known concentrations to obtain phi
        self.solve_emi()

        # update membrane potential
        phi_M_step_I = JUMP(self.phi, self.n_g)
        assign(self.phi_M_prev_PDE, pcws_constant_project(phi_M_step_I, self.Q))

        # update time
        t.assign(float(t + self.dt))

        return

    def solve_system_passive(self, Tstop, t, solver_params, membrane_params, filename=None):
        """
        Solve system with passive membrane mechanisms
        """

        # Setup solver and parameters
        self.solver_params = solver_params               # parameters for solvers
        self.direct_emi = solver_params.direct_emi       # choice of solver emi
        self.rtol_emi = solver_params.rtol_emi           # relative tolerance emi
        self.atol_emi = solver_params.atol_emi           # absolute tolerance emi
        self.threshold_emi = solver_params.threshold_emi # threshold emi

        self.splitting_scheme = False                    # no splitting scheme

        # Set filename for saving results
        self.filename = filename

        # Setup variational formulations
        self.setup_varform_emi()

        # Setup solvers
        self.setup_solver_emi()

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

        # Close files
        if filename is not None:
            self.close_h5()
            self.close_save_solver()

        # combine solution for the potential and concentrations
        uh = split(self.c_prev_k) + (self.phi,)

        return uh, self.ion_list[-1]['c']


    def solve_system_active(self, Tstop, t, solver_params, filename=None):
        """ Solve system with active membrane mechanisms (ODEs) """

        # Setup solver and parameters
        self.solver_params = solver_params               # parameters for solvers
        self.direct_emi = solver_params.direct_emi       # choice of solver emi
        self.rtol_emi = solver_params.rtol_emi           # relative tolerance emi
        self.atol_emi = solver_params.atol_emi           # absolute tolerance emi
        self.threshold_emi = solver_params.threshold_emi # threshold emi

        stimulus = self.stimulus
        stimulus_locator = self.stimulus_locator

        self.splitting_scheme = True                    # splitting scheme

        # Set filename for saving results
        self.filename = filename

        # Setup variational formulations
        self.setup_varform_emi()

        # Setup solvers
        self.setup_solver_emi()

        # Calculate ODE time step (s)
        dt_ode = float(self.dt)

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

            # solve ODEs (membrane models) for each membrane tag
            for mem_model in self.mem_models:

                ode_model = mem_model['ode']

                # Update initial values and parameters in ODE solver (based on previous PDEs step)
                ode_model.set_membrane_potential(self.phi_M_prev_PDE)
                ode_model.set_parameter('E_K', self.ion_list[0]['E'])
                ode_model.set_parameter('E_Na', self.ion_list[2]['E'])

                # set extracellular trace of K concentration at membrane
                K_e = plus(self.c_prev_k.split()[0], self.n_g)
                ode_model.set_parameter('K_e', pcws_constant_project(K_e, self.Q))

                # set intracellular trace of Na concentration at membrane
                Na_i = minus(self.ion_list[-1]['c'], self.n_g)
                ode_model.set_parameter('Na_i', pcws_constant_project(Na_i, self.Q))

                # Solve ODEs
                ode_model.step_lsoda(dt=dt_ode, \
                    stimulus=stimulus, stimulus_locator=stimulus_locator)

                # Update PDE functions based on ODE output
                ode_model.get_membrane_potential(self.phi_M_prev_PDE)

                # Update src terms for next PDE step based on ODE output
                for ion, I_ch_k in mem_model['I_ch_k'].items():
                    # update src term for each ion species
                    ode_model.get_parameter("I_ch_" + ion, I_ch_k)

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

        # Close files
        if filename is not None:
            self.close_h5()
            self.close_save_solver()

        return



    def initialize_solver_savefile(self, path_timings):
        """ write CPU timings (solve and assemble), condition number and number of
            iterations to file """

        # if directory does not exists, create it
        if not os.path.exists(path_timings):
            os.mkdir(path_timings)

        #path_timings = "results/it_count_2D/"
        reso = self.solver_params.resolution

        # get number of mesh cells, number of dofs emi, number of dofs knp
        num_cells = self.mesh.num_cells()
        dofs_emi = self.V_emi.dim()
        dofs_knp = self.V_knp.dim()

        if self.direct_emi:
            self.file_emi_solve = open(path_timings + "emi_solve_dir_%d.txt" % reso, "w")
            self.file_emi_assem = open(path_timings + "emi_assem_dir_%d.txt" % reso, "w")
            self.file_emi_niter = None
        else:
            self.file_emi_solve = open(path_timings + "emi_solve_%d.txt" % reso, "w")
            self.file_emi_assem = open(path_timings + "emi_assem_%d.txt" % reso, "w")
            self.file_emi_niter = open(path_timings + "emi_niter_%d.txt" % reso, "w")

            self.file_emi_niter.write("num cells: %d \n" % num_cells)
            self.file_emi_niter.write("dofs: %d \n" % dofs_emi)

        self.file_emi_solve.write("num cells: %d \n" % num_cells)
        self.file_emi_solve.write("dofs: %d \n" % dofs_emi)
        self.file_emi_assem.write("num cells: %d \n" % num_cells)
        self.file_emi_assem.write("dofs: %d \n" % dofs_emi)

        return

    def close_save_solver(self):

        if not self.direct_emi:
            self.file_emi_niter.close()

        #self.file_phi_M_1.close()
        self.file_emi_solve.close()
        self.file_emi_assem.close()

        return


    def initialize_h5_savefile(self, filename):
        """ initialize h5 file """
        self.h5_idx = 0
        self.h5_file = HDF5File(self.mesh.mpi_comm(), filename, 'w')
        self.h5_file.write(self.mesh, '/mesh')
        self.h5_file.write(self.subdomains, '/subdomains')
        self.h5_file.write(self.surfaces, '/surfaces')

        self.h5_file.write(self.c_prev_k, '/concentrations',  self.h5_idx)
        self.h5_file.write(self.ion_list[-1]['c'], '/elim_concentration',  self.h5_idx)
        self.h5_file.write(self.phi, '/potential', self.h5_idx)

        return

    def save_h5(self):
        """ save results to h5 file """
        self.h5_idx += 1
        self.h5_file.write(self.c_prev_k, '/concentrations',  self.h5_idx)
        self.h5_file.write(self.ion_list[-1]['c'], '/elim_concentration',  self.h5_idx)
        self.h5_file.write(self.phi, '/potential', self.h5_idx)

        return

    def close_h5(self):
        """ close h5 file """
        self.h5_file.close()
        return

    def make_global(self, f):

        mesh = self.mesh
        subdomains = self.subdomains

        # DG space for projecting coefficients
        Q = FunctionSpace(self.mesh, self.PK_knp)

        dofmap = Q.dofmap()

        # list of list of dofs for each sub-domain
        o_dofss = {key: [] for key in f.keys()}

        i = 0
        # fill list with relevant dofs for each sub-domains
        for cell in cells(mesh):
            for tag, function in f.items():
                if subdomains[cell] == tag:
                    o_dofss[tag].extend(dofmap.cell_dofs(cell.index()))
                    i += 1
                    break

        # check that all cells are matched with tags in loop above
        assert sum(1 for _ in cells(mesh)) == i, \
               "Dictionaries for DG data must match cell tags in mesh"

        # set dofs list
        for key, o_dofs in o_dofss.items():
            o_dofs = list(set(o_dofs))

        F = Function(Q)

        for tag, function in f.items():
            # interpolate data for sub-domain with tag tag
            F_tag = interpolate(function, Q)
            # copy to global function
            F.vector()[o_dofss[tag]] = F_tag.vector()[o_dofss[tag]]

        return F
