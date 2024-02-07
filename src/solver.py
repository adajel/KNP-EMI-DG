from dolfin import *
import numpy as np
import sys
import os
import time
from petsc4py import PETSc

from utils import pcws_constant_project
from utils import interface_normal, plus, minus
from utils import CellCenterDistance

from memory_profiler import profile

#from membrane import MembraneModel
#import hh_model as ode

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
#   equation, yielding the following system: Given c_a_ and c_b_,
#   iterate over the two following steps:
#
#       step I:  (emi) find phi by solving (2), with J^a(c_a_, phi) and J^b(c_b_, phi)
#       step II: (knp) find c_a and c_b by solving (1) with J^k(c_a, c_b, phi_), where phi_
#                is the solution from step I
#     (step III: solve ODEs at interface, and update membrane potential)
#
# Membrane potential is defined as phi_i - phi_e, since we have
# marked cell in ECS with 2 and cells in ICS with 1 we have an
# interface normal pointing inwards
#    ____________________
#   |                    |
#   |      ________      |
#   |     |        |     |
#   | ECS |   ICS  |     |
#   |  2  |->  1   |     |
#   | (+) |   (-)  |     |
#   |     |________|     |
#   |                    |
#   |____________________|
#
# Normal will always point from higher to lower (e.g. from 2 -> 1)
#
# NB! The code assumes all membrane facets are tagged with 1, and that all interior
# facets are tagged with 0.

class Solver:
    def __init__(self, params, ion_list, degree_emi=1, degree_knp=1, mms=None):
        """ initialize solver """

        self.ion_list = ion_list            # list of ions species
        self.N_ions = len(ion_list[:-1])    # number of ions to solver for
        self.degree_emi = degree_emi        # polynomial degree
        self.degree_knp = degree_knp        # polynomial degree
        self.mms = mms                      # boolean for mms test
        self.params = params                # physical parameters

        # frequency for saving results
        self.sf = 1

        # timers
        self.ode_solve_timer = 0
        self.emi_solve_timer = 0
        self.knp_solve_timer = 0
        self.emi_ass_timer = 0
        self.knp_ass_timer = 0

        return

    def setup_domain(self, mesh, subdomains, surfaces):
        """ setup mesh and associated parameters, and element spaces """

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
        self.hA_knp = CellDiameter(self.mesh)
        self.hA_emi = CellDiameter(self.mesh)

        # interface normal
        self.n_g = interface_normal(subdomains, mesh)

        # DG penalty parameter
        self.gdim = self.mesh.geometry().dim()
        self.tau_emi = Constant(20*self.gdim*self.degree_emi)
        self.tau_knp = Constant(20*self.gdim*self.degree_knp)

        # DG0 for EMI sub-problem
        if self.degree_emi == 0:
            self.hA_emi = CellCenterDistance(self.mesh)
            self.tau_emi = 1.0

        # DG elements for ion concentrations and the potential
        self.PK_emi = FiniteElement('DG', mesh.ufl_cell(), self.degree_emi)
        self.PK_knp = FiniteElement('DG', mesh.ufl_cell(), self.degree_knp)

        # For the MMS problem, we need unique tags for each of the interface walls
        if self.mms is not None:
            self.lm_tags = [1, 2, 3, 4]

        return


    def setup_parameters(self):
        """ setup physical parameters """

        # setup physical parameters
        params = self.params

        # get physical parameters
        self.C_phi = Constant(params.C_phi)             # coupling coefficient
        self.C_M = Constant(params.C_M)                 # capacitance
        self.dt = Constant(params.dt)                   # time step
        self.F = Constant(params.F)                     # Faraday constant
        self.R = Constant(self.params.R)                # Gas constant
        self.temperature = Constant(params.temperature) # temperature
        #self.phi_M_init = Constant(params.phi_M_init)  # initial membrane potential
        self.phi_M_init = params.phi_M_init             # initial membrane potential
        self.psi = self.F/(self.R*self.temperature)     # shorthand

        # define global diffusion coefficients
        for idx, ion in enumerate(self.ion_list):

            # project diffusion coefficients to PK based on subdomain
            D = self.make_global(Constant(ion['D1']), Constant(ion['D2']))
            ion['D'] = D

            # define global coupling coefficient (for MMS case)
            if self.mms is not None:
                # project coupling coefficients to PK based on subdomain
                C = self.make_global(Constant(ion['C1']), Constant(ion['C2']))
                ion['C'] = C

        return


    def setup_FEM_spaces(self):
        """ setup function spaces and functions """

        # create function space for potential (phi)
        self.V_emi = FunctionSpace(self.mesh, self.PK_emi)
        # function for solution potential
        self.phi = Function(self.V_emi)

        # set up finite element space for concentrations (c)
        ME = MixedElement([self.PK_knp]*self.N_ions)
        self.V_knp = FunctionSpace(self.mesh, ME)

        # function for solution concentrations
        self.c = Function(self.V_knp)
        # function for previous solution concentrations in time stepping
        self.c_prev_n = Function(self.V_knp)
        # function for previous solution concentrations in Picard iteration
        self.c_prev_k = Function(self.V_knp)

        ion_list = self.ion_list

        #print("dofs EMI", self.V_emi.dim())
        #print("dofs KNP", self.V_knp.dim())
        #print("num cells", self.mesh.num_cells())
        #print("sum dofs", self.V_emi.dim() + self.V_knp.dim())
        #sys.exit(0)

        # set initial conditions concentrations
        for idx, ion in enumerate(self.ion_list):
            # get initial condition for ions
            c1 = ion['c1_init']; c2 = ion['c2_init']

            # interpolate initial conditions to global function and assign
            c_init = self.make_global(c1, c2)

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
        self.kappa = 0                      # kappa
        a = 0; L = 0                        # rhs and lhs forms
        self.I_ch = [0]*len(self.mem_tags) # sum of ion specific channel currents
        self.alpha_sum = 0                 # sum of fractions intracellular

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

            # update ion specific channel current (for each membrane model)
            for jdx, mm in enumerate(ion['mem_models']):
                mm['I_ch'] = mm['g_k']*(self.phi_M_prev_PDE - ion['E'])
                # update total channel current for each tag
                self.I_ch[jdx] += mm['I_ch']

            # update alpha
            self.alpha_sum += ion['D'] * ion['z'] * ion['z'] * c_k_

            # global kappa
            self.kappa += F * ion['z'] * ion['z'] * ion['D'] * psi * c_k_

            # Add terms rhs (diffusive terms)
            L += - F * ion['z'] * inner((ion['D'])*grad(c_k_), grad(v_phi)) * dx \
                 + F * ion['z'] * inner(dot(avg((ion['D'])*grad(c_k_)), n('+')), jump(v_phi)) * dS(0) \

        # equation potential (drift terms)
        a += inner(self.kappa*grad(u_phi), grad(v_phi)) * dx \
           - inner(dot(avg(self.kappa*grad(u_phi)), n('+')), jump(v_phi)) * dS(0) \
           - inner(dot(avg(self.kappa*grad(v_phi)), n('+')), jump(u_phi)) * dS(0) \
           + tau_emi/avg(hA) * inner(avg(self.kappa)*jump(u_phi), jump(v_phi)) * dS(0)

        if self.mms is None:

            # coupling condition at interface
            if self.splitting_scheme:
                # robin condition with splitting
                g_robin_emi = [self.phi_M_prev_PDE]*len(self.mem_tags)
            else:
                # original robin condition (without splitting)
                # TODO
                #g_robin_emi = self.phi_M_prev_PDE - (1 / C_phi) * self.I_ch
                g_robin_emi = [self.phi_M_prev_PDE - (1 / C_phi) * I for I in self.I_ch]

            for jdx, tag in enumerate(self.mem_tags):
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
               + inner(fphi2, v_phi)*dx(2) \

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
            opts.setValue('ksp_view', None)

            # tolerance seems to depend on dimension of mesh
            if self.gdim == 3:
                opts.setValue('ksp_rtol', 1E-5)
                opts.setValue('ksp_atol', 1E-13)
                opts.setValue('pc_hypre_boomeramg_strong_threshold', 0.9)
            elif self.gdim == 2:
                opts.setValue('ksp_rtol', 1E-5)
                opts.setValue('ksp_atol', 1E-13)
                #opts.setValue('ksp_rtol', 1E-7)
                #opts.setValue('ksp_atol', 1E-14)
                opts.setValue('pc_hypre_boomeramg_strong_threshold', 0.9)

            opts.setValue('ksp_initial_guess_nonzero', 1)
            opts.setValue('ksp_converged_reason', None)
            opts.setValue('pc_type', 'hypre')

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

    #@profile
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

            print("-------------------------")
            print("niter emi", niter)
            print("-------------------------")

        if self.filename is not None:
            self.file_emi_solve.write("solve_time: %.4f \n" % (res))

        # The solution x lives in mixed space, but we only want part of the
        # solution -> fill phi with relevant part of solution.
        # self.phi.assign(project(self.x_emi.split()[0], self.V_emi))

        self.phi.vector().vec().array_w[:] = self.x_emi.array_r[:]
        # make assign above work in parallel
        self.phi.vector().vec().ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        return


    def setup_varform_knp(self):
        """ setup variational form for the knp system """

        dx = self.dx; ds = self.ds; dS = self.dS    # measures
        n = self.n; hA = self.hA_knp; n_g = self.n_g    # facet area and normal
        tau_knp = self.tau_knp                      # penalty parameter
        ion_list = self.ion_list                    # ion list
        psi = self.psi; C_phi = self.C_phi          # physical parameters
        C_M = self.C_M; F = self.F                  # physical parameters
        phi = self.phi                              # potential

        us = TrialFunctions(self.V_knp)
        vs = TestFunctions(self.V_knp)

        # initialize form
        a = 0; L = 0

        for idx, ion in enumerate(ion_list[:-1]):
            # get trial and test functions
            u_c = us[idx]
            v_c = vs[idx]

            # get previous concentration
            c_n_ = split(self.c_prev_n)[idx]
            c_k_ = split(self.c_prev_k)[idx]

            # get valence and diffusion coefficients
            z = ion['z']; D = ion['D']

            # upwinding: We first define function un returning:
            #       dot(u,n)    if dot(u, n) >  0
            #       0           if dot(u, n) <= 0
            #
            # We would like to upwind s.t.
            #   c('+') is chosen if dot(u, n('+')) > 0,
            #   c('-') is chosen if dot(u, n('+')) < 0.
            #
            # The expression:
            #       un('+')*c('+') - un('-')*c('-') = jump(un*c)
            #
            # give this. Given n('+') = -n('-'), we have that:
            #   dot(u, n('+')) > 0
            #   dot(u, n('-')) < 0.
            # As such, if dot(u, n('+')) > 0, we have that:
            #   un('+') is dot(u, n), and
            #   un('-') = 0.
            # and the expression above becomes un('+')*c('+') - 0*c('-') =
            # un('+')*c('+').

            # define upwind help function
            un = 0.5*(dot(D * grad(phi), n) + abs(dot(D * grad(phi), n)))

            # equation ion concentration diffusive term with SIP (symmetric)
            a += 1.0/self.dt * u_c * v_c * dx \
               + inner(D * grad(u_c), grad(v_c)) * dx \
               - inner(dot(avg(D * grad(u_c)), n('+')), jump(v_c)) * dS(0) \
               - inner(dot(avg(D * grad(v_c)), n('+')), jump(u_c)) * dS(0) \
               + tau_knp/avg(hA) * inner(jump(D * u_c), jump(v_c)) * dS(0)

            # drift (advection) terms + upwinding
            a += + z * psi * inner(D * u_c * grad(phi), grad(v_c)) * dx \
                 - z * psi * jump(v_c) * jump(un * u_c) * dS(0)

            # add terms for approximating time derivative
            L += 1.0/self.dt * c_n_ * v_c * dx

            if self.mms is None:
                # calculate alpha
                alpha = D * z * z * c_k_ / self.alpha_sum

                # calculate coupling coefficient
                C = alpha * C_M / (F * z * self.dt)

                # loop through each membrane model (and associated membrane tag)
                for jdx, mm in enumerate(ion['mem_models']):
                    if self.splitting_scheme:
                        # robin condition with splitting
                        g_robin_knp = self.phi_M_prev_PDE \
                                    - self.dt / (C_M * alpha) * mm['I_ch'] \
                                    + (self.dt / C_M) * self.I_ch[jdx]
                    else:
                        # original robin condition (without splitting)
                        g_robin_knp = self.phi_M_prev_PDE \
                                    - self.dt / (C_M * alpha) * mm['I_ch']

                    # add coupling condition at interface
                    L += JUMP(C * g_robin_knp * v_c, self.n_g) * dS(mm['tag'])

                    # add coupling terms on interface gamma
                    L += - jump(phi) * jump(C) * avg(v_c) * dS(mm['tag']) \
                         - jump(phi) * avg(C) * jump(v_c) * dS(mm['tag'])

            # add terms for manufactured solutions test
            if self.mms is not None:
                # get mms data
                fc1 = ion['f1']
                fc2 = ion['f2']
                g_robin_knp_1 = ion['g_robin_1']
                g_robin_knp_2 = ion['g_robin_2']

                # get global coupling coefficients
                C = ion['C']; C_1 = ion['C1']; C_2 = ion['C2']

                lm_tags = self.lm_tags

                # MMS specific: add source terms
                L += inner(fc1, v_c)*dx(1) \
                   + inner(fc2, v_c)*dx(2) \

                # coupling terms on interface gamma
                L += - sum(jump(phi) * jump(C) * avg(v_c) * dS(tag) for tag in lm_tags) \
                     - sum(jump(phi) * avg(C) * jump(v_c) * dS(tag) for tag in lm_tags)

                # define robin condition on interface gamma
                L += sum(inner(C_1 * g_robin_knp_1[tag], minus(v_c, n_g)) * dS(tag) for tag in lm_tags) \
                   - sum(inner(C_2 * g_robin_knp_2[tag], plus(v_c, n_g)) * dS(tag) for tag in lm_tags)

                # MMS specific: add neumann contribution
                L += - dot(ion['bdry'], n) * v_c * ds

        # set forms lhs (A) and rhs (L)
        self.A_knp = a
        self.L_knp = L

        return

    def setup_solver_knp(self):
        """ setup KSP solver for KNP sub-problem """

        # create solver
        ksp = PETSc.KSP().create()

        if self.direct_knp:
            # set options direct solver
            opts = PETSc.Options("KNP_DIR")  # get options
            opts["mat_mumps_icntl_4"] = 1    # set amount of info output
            opts["mat_mumps_icntl_14"] = 40  # set percentage of ???

            pc = ksp.getPC()                 # get pc
            pc.setType("lu")                 # set solver to LU
            pc.setFactorSolverType("mumps")  # set LU solver to use mumps
            ksp.setOptionsPrefix("KNP_DIR")
            ksp.setFromOptions()             # update ksp with options set above
        else:
            # set options iterative solver
            opts = PETSc.Options('KNP_ITER')
            opts.setValue('ksp_type', 'gmres')
            opts.setValue('ksp_min_it', 5)
            opts.setValue("ksp_max_it", 1000)
            opts.setValue('pc_type', 'hypre')
            opts.setValue("ksp_converged_reason", None)

            # tolerance seems to depend on dimension of mesh
            if self.gdim == 3:
                opts.setValue('ksp_rtol', 1E-7)
                opts.setValue('ksp_atol', 2E-17)
                opts.setValue('pc_hypre_boomeramg_strong_threshold', 0.75)
            elif self.gdim == 2:
                opts.setValue('ksp_rtol', 1E-7)
                opts.setValue('ksp_atol', 1E-40)

            opts.setValue("ksp_initial_guess_nonzero", 1)
            opts.setValue("ksp_view", None)
            opts.setValue("ksp_monitor_true_residual", None)

            ksp.setOptionsPrefix('KNP_ITER')
            ksp.setFromOptions()

        # set knp solver
        self.ksp_knp = ksp

        # timer start
        ts = time.perf_counter()

        # assemble
        AA_knp, bb_knp = map(assemble, (self.A_knp, self.L_knp))
        self.AA_knp = as_backend_type(AA_knp)
        self.bb_knp = as_backend_type(bb_knp)

        self.x_knp, _ = self.AA_knp.mat().createVecs()
        self.x_knp.axpy(1, as_backend_type(self.c.vector()).vec())

        # timer end
        te = time.perf_counter()
        self.knp_ass_timer +=  te - ts

        return

    #@profile
    def solve_knp(self):
        """ solve knp system """

        # timer start
        ts = time.perf_counter()

        # reassemble matrices and vector
        assemble(self.A_knp, self.AA_knp)
        assemble(self.L_knp, self.bb_knp)

        # convert matrices and vector
        AA_knp = self.AA_knp.mat()
        bb_knp = self.bb_knp.vec()
        x_knp = self.x_knp

        # solution vector
        #x_knp *= 0.
        #x_knp.axpy(1, as_backend_type(self.c.vector()).vec())

        # timer end
        te = time.perf_counter()
        res = te - ts
        print(f"{bcolors.OKGREEN} CPU Execution time PDE assemble knp: {res:.4f} seconds {bcolors.ENDC}")
        self.knp_ass_timer += res

        # write assembly time to file
        if self.filename is not None:
            self.file_knp_assem.write("ass_time: %.4f \n" % (res))

        if self.direct_knp:
            # set operators
            AA_knp.convert(PETSc.Mat.Type.AIJ)
            self.ksp_knp.setOperators(AA_knp, AA_knp)

            # solve knp system with direct solver
            ts = time.perf_counter()            # timer start
            self.ksp_knp.solve(bb_knp, x_knp)   # solve
            te = time.perf_counter()            # timer end

            # print and write CPU solve time to file
            res = te - ts
            print(f"{bcolors.OKGREEN} CPU Execution time PDE solve knp: {res:.4f} seconds {bcolors.ENDC}")
            self.knp_solve_timer += res

            if self.filename is not None:
                self.file_knp_solve.write("solve_time: %.4f \n" % (res))
        else:
            # set operators
            self.ksp_knp.setOperators(AA_knp, AA_knp)

            # solve the knp system with iterative solver
            ts = time.perf_counter()            # timer start
            self.ksp_knp.solve(bb_knp, x_knp)   # solve system
            te = time.perf_counter()            # timer end

            # print and write CPU solve time to file
            res = te - ts
            print(f"{bcolors.OKGREEN} CPU Execution time PDE solve knp: {res:.4f} seconds {bcolors.ENDC}")
            self.knp_solve_timer += res

            # print and write number of iterations
            niters = self.ksp_knp.getIterationNumber()

            if self.filename is not None:
                self.file_knp_solve.write("solve_time: %.4f \n" % (res))
                self.file_knp_niter.write("niter: %d \n" % niters)

            print("-------------------------")
            print("niter knp", niters)
            print("-------------------------")

        # assign new value to function c
        self.c.vector().vec().array_w[:] = x_knp.array_r[:]
        # make assign above work in parallel
        self.c.vector().vec().ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        return


    def solve_for_time_step(self, k, t):
        """ solve system for one global time step dt"""

        print("------------------------------------------------")
        print(f"{bcolors.WARNING} t = {float(t)} {bcolors.ENDC}")
        print(f"{bcolors.WARNING} k = {k} {bcolors.ENDC}")
        print("------------------------------------------------")

        # Step I: solve emi equations with known concentrations to obtain phi
        self.solve_emi()

        # Step II: solve knp equations with known phi to obtain concentrations
        self.solve_knp()

        # update previous concentrations
        self.c_prev_k.assign(self.c)
        self.c_prev_n.assign(self.c)

        # update membrane potential
        phi_M_step_I = JUMP(self.phi, self.n_g)
        assign(self.phi_M_prev_PDE, pcws_constant_project(phi_M_step_I, self.Q))

        # get physical parameters
        F = self.F; R = self.R; temperature = self.temperature

        # variable for eliminated ion concentration
        c_elim = 0

        ion_list = self.ion_list

        # update Nernst potentials for next global time level
        for idx, ion in enumerate(ion_list[:-1]):
            # get current solution concentration
            c_k_ = split(self.c_prev_k)[idx]
            # update Nernst potential
            E = R * temperature / (F * ion['z']) * ln(plus(c_k_, self.n_g) / minus(c_k_, self.n_g))
            ion['E'].assign(pcws_constant_project(E, self.Q))

            # add ion specific contribution to eliminated ion concentration
            c_elim += - (1.0 / ion_list[-1]['z']) * ion['z'] * c_k_

        # update eliminated ion concentration
        self.ion_list[-1]['c'].assign(project(c_elim, \
                self.V_knp.sub(self.N_ions - 1).collapse()))

        # update Nernst potential for eliminated ion
        E = R * temperature / (F * ion_list[-1]['z']) * ln(plus(ion_list[-1]['c'], self.n_g) / minus(ion_list[-1]['c'], self.n_g))
        ion_list[-1]['E'].assign(pcws_constant_project(E, self.Q))

        # update time
        t.assign(float(t + self.dt))

        return


    def solve_for_time_step_picard(self, k, t):
        """ solve system for one global time step using Picard iterations """

        print("------------------------------------------------")
        print(f"{bcolors.WARNING} t = {float(t)} {bcolors.ENDC}")
        print(f"{bcolors.WARNING} k = {k} {bcolors.ENDC}")
        print("------------------------------------------------")

        # update time
        t.assign(float(t + self.dt))

        # define picard parameters
        tol = 1.0e-4    # tolerance
        eps = 2.0       # set eps bigger than tolerance initially
        max_iter = 25   # max number of iterations
        iter = 0        # counter for number of iterations

        # inner Picard iteration to solve PDEs
        while eps > tol:

            iter += 1

            # solve emi equation for potential with previous concentrations
            self.solve_emi()

            # solve knp equations for concentrations with known phi
            self.solve_knp()

            # calculate diff between current and previous Picard iteration
            diff = self.c_prev_k.vector() - self.c.vector()
            eps = np.linalg.norm(diff, ord=np.Inf)

            # update previous concentrations for next Picard level
            self.c_prev_k.assign(self.c)

            # get physical parameters
            F = self.F; R = self.R; temperature = self.temperature

            c_elim = 0

            ion_list = self.ion_list

            # update Nernst potentials for next Picard level
            for idx, ion in enumerate(ion_list[:-1]):
                # get current solution concentration
                c_k_ = split(self.c_prev_k)[idx]
                # update Nernst potential
                E = R * temperature / (F * ion['z']) * ln(plus(c_k_, self.n_g) / minus(c_k_, self.n_g))
                ion['E'].assign(pcws_constant_project(E, self.Q))

                # add ion specific contribution to eliminated ion concentration
                c_elim += - (1.0 / ion_list[-1]['z']) * ion['z'] * c_k_

            # update eliminated ion concentration for next Picard level
            ion_list[-1]['c'].assign(project(c_elim, \
            self.V_knp.sub(self.N_ions - 1).collapse()))

            # update Nernst potential for eliminated ion
            E = R * temperature / (F * ion_list[-1]['z']) * ln(plus(ion_list[-1]['c'], self.n_g) / minus(ion_list[-1]['c'], self.n_g))
            ion_list[-1]['E'].assign(pcws_constant_project(E, self.Q))

            # exit if iteration exceeds maximum number of iterations
            if iter > max_iter:
                print("Picard solver diverged")
                sys.exit(2)

        # update previous concentrations for next global time step
        self.c_prev_n.assign(self.c_prev_k)

        # update membrane potential
        phi_M_step_I = JUMP(self.phi, self.n_g)
        assign(self.phi_M_prev_PDE, pcws_constant_project(phi_M_step_I, self.Q))

        # print Picard output
        print(f"{bcolors.OKCYAN} Summary Picard: eps = {eps},, #iters = {iter} {bcolors.ENDC}")

        return


    def solve_system_passive(self, Tstop, t, solver_params, membrane_params, filename=None):
        """ solve system with passive membrane mechanisms """

        # parameters for solvers
        self.solver_params = solver_params

        g_a = membrane_params.g_a_leak       # Na max conductivity (S/m**2)
        g_b = membrane_params.g_b_leak       # K max conductivity (S/m**2)
        g_c = membrane_params.g_c_leak       # Na leak conductivity (S/m**2)

        # Set membrane models and associated tags
        mem_models_a = [{'tag':1, 'g_k':g_a}]
        mem_models_b = [{'tag':1, 'g_k':g_b}]
        mem_models_c = [{'tag':1, 'g_k':g_c}]

        # Set tags (NB! Must match with tags in membrane models)
        self.mem_tags = [1]

        # Assign membrane models
        self.ion_list[0]['mem_models'] = mem_models_a
        self.ion_list[1]['mem_models'] = mem_models_b
        self.ion_list[2]['mem_models'] = mem_models_c

        # Set filename for saving results
        self.filename = filename

        # choice of solver
        self.direct_emi = solver_params.direct_emi
        self.direct_knp = solver_params.direct_knp

        # splitting scheme
        self.splitting_scheme = False

        # setup physical parameters
        self.setup_parameters()
        # setup function spaces and numerical parameters
        self.setup_FEM_spaces()

        # setup variational formulation
        self.setup_varform_emi()
        self.setup_varform_knp()

        self.setup_solver_emi()
        self.setup_solver_knp()

        # initialize save results
        if filename is not None:
            self.initialize_h5_savefile(filename + '/results.h5')

        for k in range(int(round(Tstop/float(self.dt)))):

            self.solve_for_time_step(k, t)
            #self.solve_for_time_step_picard(k, t)

            # save to file
            if (k % self.sf) == 0 and filename is not None:
                # save results
                self.save_h5()

        # combine solution for the potential and concentrations
        uh = split(self.c) + (self.phi,)

        if not self.direct_emi and filename is not None:
            self.file_emi_niter.close()
            self.file_knp_niter.close()

        # initialize save results
        if filename is not None:
            self.close_h5()

        return uh, self.ion_list[-1]['c']


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

        print("dofs")
        print(dofs_knp)
        print(dofs_emi)
        print(num_cells)
        print("dofs")

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

        if self.direct_knp:
            self.file_knp_solve = open(path_timings + "knp_solve_dir_%d.txt" % reso, "w")
            self.file_knp_assem = open(path_timings + "knp_assem_dir_%d.txt" % reso, "w")
            self.file_knp_niter = None
        else:
            self.file_knp_solve = open(path_timings + "knp_solve_%d.txt" % reso, "w")
            self.file_knp_assem = open(path_timings + "knp_assem_%d.txt" % reso, "w")
            self.file_knp_niter = open(path_timings + "knp_niter_%d.txt" % reso, "w")

            self.file_knp_niter.write("num cells: %d \n" % num_cells)
            self.file_knp_niter.write("dofs: %d \n" % dofs_knp)

        self.file_knp_solve.write("num cells: %d \n" % num_cells)
        self.file_knp_solve.write("dofs: %d \n" % dofs_knp)
        self.file_knp_assem.write("num cells: %d \n" % num_cells)
        self.file_knp_assem.write("dofs: %d \n" % dofs_knp)

        # open files for saving bulk results
        self.f_pot = File('results/active_knp/pot.pvd')
        self.f_pot_grad = File('results/active_knp/pot_grad.pvd')
        self.f_Na = File('results/active_knp/Na.pvd')
        self.f_K = File('results/active_knp/K.pvd')
        self.f_Cl = File('results/active_knp/Cl.pvd')

        self.f_grad_Na = File('results/active_knp/Na_grad.pvd')
        self.f_grad_K = File('results/active_knp/K_grad.pvd')

        self.f_kappa = File('results/active_knp/Kappa.pvd')

        return

    def save_solver(self, k):
            # just for debugging
            phi = self.phi

            VDG1 = VectorFunctionSpace(self.mesh, "DG", 0)
            VDG0 = FunctionSpace(self.mesh, "DG", 0)

            phi_ = project(grad(phi), VDG1)
            K_ = project(self.c.split()[0], VDG0)
            Cl_ = project(self.c.split()[1], VDG0)
            Na_ = project(self.ion_list[-1]['c'], VDG0)

            grad_Na_ = project(grad(self.c.split()[0]), VDG1)
            grad_K_ = project(grad(self.c.split()[1]), VDG1)
            self.f_pot_grad << (phi_, k)
            self.f_pot << (phi, k)
            self.f_Na << (Na_, k)
            self.f_K << (K_, k)
            self.f_Cl << (Cl_, k)

            self.f_grad_Na << (grad_Na_, k)
            self.f_grad_K << (grad_K_, k)

            self.f_kappa << (project(self.kappa, VDG0), k)

            return

    def close_save_solver(self):

        if not self.direct_emi:
            self.file_emi_niter.close()
            self.file_knp_niter.close()

        #self.file_phi_M_1.close()
        self.file_emi_solve.close()
        self.file_knp_solve.close()
        self.file_emi_assem.close()
        self.file_knp_assem.close()

        return


    def initialize_h5_savefile(self, filename):
        """ initialize h5 file """
        self.h5_idx = 0
        self.h5_file = HDF5File(self.mesh.mpi_comm(), filename, 'w')
        self.h5_file.write(self.mesh, '/mesh')
        self.h5_file.write(self.subdomains, '/subdomains')
        self.h5_file.write(self.surfaces, '/surfaces')

        self.h5_file.write(self.c, '/concentrations',  self.h5_idx)
        self.h5_file.write(self.ion_list[-1]['c'], '/elim_concentration',  self.h5_idx)
        self.h5_file.write(self.phi, '/potential', self.h5_idx)

        self.h5_file.write(self.n_HH, '/n_HH',  self.h5_idx)
        self.h5_file.write(self.h_HH, '/h_HH',  self.h5_idx)
        self.h5_file.write(self.m_HH, '/m_HH',  self.h5_idx)

        return

    def save_h5(self):
        """ save results to h5 file """
        self.h5_idx += 1
        self.h5_file.write(self.c, '/concentrations',  self.h5_idx)
        self.h5_file.write(self.ion_list[-1]['c'], '/elim_concentration',  self.h5_idx)
        self.h5_file.write(self.phi, '/potential', self.h5_idx)

        self.h5_file.write(self.n_HH, '/n_HH',  self.h5_idx)
        self.h5_file.write(self.h_HH, '/h_HH',  self.h5_idx)
        self.h5_file.write(self.m_HH, '/m_HH',  self.h5_idx)

        return

    def close_h5(self):
        """ close h5 file """
        self.h5_file.close()
        return

    def make_global(self, f1, f2):

        mesh = self.mesh
        subdomains = self.subdomains

        # DG0 space for projecting coefficients
        Q = FunctionSpace(self.mesh, self.PK_knp)

        dofmap = Q.dofmap()

        # dofs in domain 1 and domain 2
        o1_dofs = []
        o2_dofs = []

        for cell in cells(mesh): # compute dofs in the domains
            if subdomains[cell] == 1:
                o1_dofs.extend(dofmap.cell_dofs(cell.index()))
            elif subdomains[cell] == 2:
                o2_dofs.extend(dofmap.cell_dofs(cell.index()))
            else:
                print("cell not marked")
                sys.exit(0)

        o2_dofs = list(set(o2_dofs))
        o1_dofs = list(set(o1_dofs))

        F1 = interpolate(f1, Q)
        F2 = interpolate(f2, Q)
        F = Function(Q)

        F.vector()[o2_dofs] = F2.vector()[o2_dofs]
        F.vector()[o1_dofs] = F1.vector()[o1_dofs]

        return F
