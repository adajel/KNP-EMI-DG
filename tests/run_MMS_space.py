from dolfin import *
import numpy as np
import sys
import os

import matplotlib.pyplot as plt

parameters['ghost_mode'] = 'shared_vertex'

if __name__ == '__main__':
    from mms_space import setup_mms
    from knpemidg import Solver
    from collections import namedtuple
    from itertools import chain

    GREEN = '\033[1;37;32m%s\033[0m'

    dt_0 = 1.0e-10
    Tstop = dt_0*2      # end time

    degree = 2

    hs, errors_ca, errors_cb, errors_phi = [], [], [], []
    rates_ca, rates_cb, rates_phi = [], [], []

    i = 0
    for resolution in range(2, 8):

        t = Constant(0.0)    # time constant (s)
        dt = dt_0

        D_a1 = Constant(6); D_a2 = Constant(5)
        D_b1 = Constant(3); D_b2 = Constant(4)
        D_c1 = Constant(1); D_c2 = Constant(2)

        C_a1 = Constant(1); C_a2 = Constant(2)
        C_b1 = Constant(2); C_b2 = Constant(4)
        C_c1 = Constant(3); C_c2 = Constant(2)

        z_a = Constant(-1.0); z_b = Constant(1.0); z_c = Constant(1.0)

        F = Constant(1); C_M = Constant(1.0);  R = Constant(1); temperature = Constant(1)

        phi_M_init = Expression('sin(2*pi*(x[0]-x[1])) - cos(pi*(x[0]+x[1]))', degree=4)

        # Set parameters
        params = namedtuple('params', (
        'D_a1', 'D_a2', 'D_b1', 'D_b2', 'D_c1', 'D_c2',\
        'C_a1', 'C_a2', 'C_b1', 'C_b2', 'C_c1', 'C_c2',\
        'C_phi', \
        'z_a', 'z_b', 'z_c', 'dt', 'F', 'C_M', 'phi_M_init',\
        'R', 'temperature'))(D_a1, D_a2, D_b1, D_b2, D_c1, D_c2,\
                                   C_a1, C_a2, C_b1, C_b2, C_c1, C_c2,\
                                   C_M/dt, \
                                   z_a, z_b, z_c, dt, F, C_M, phi_M_init, \
                                   R, temperature)

        t = Constant(0.0)

        # get mesh, subdomains, surfaces path
        here = os.path.abspath(os.path.dirname(__file__))
        mesh_prefix = os.path.join(here, 'meshes/MMS/')
        mesh_path = mesh_prefix + 'mesh_' + str(resolution) + '.xml'
        subdomains_path = mesh_prefix + 'subdomains_' + str(resolution) + '.xml'
        surfaces_path = mesh_prefix + 'surfaces_' + str(resolution) + '.xml'

        # generate mesh if it does not exist
        if not os.path.isfile(mesh_path):
            from make_mesh_MMS import main
            main(["-r", str(resolution), "-d", mesh_prefix])

        mesh = Mesh(mesh_path)
        subdomains = MeshFunction('size_t', mesh, subdomains_path)
        surfaces = MeshFunction('size_t', mesh, surfaces_path)

        mms = setup_mms(params, t, mesh)

        ca2 = mms.solution['c_a2']
        cb2 = mms.solution['c_b2']
        cc2 = mms.solution['c_c2']
        phi2 = mms.solution['phi_2']
        ca1 = mms.solution['c_a1']
        cb1 = mms.solution['c_b1']
        cc1 = mms.solution['c_c1']
        phi1 = mms.solution['phi_1']

        # Recall our geometry is
        #      ______
        #     [      ]
        #     [  ()  ]
        #     [______]
        #
        # Wher () is the surfaces['inner']. On the outer surfaces we prescribe
        # Neumann bcs.

        # get initial concentrations
        ca1_init = mms.solution['c_a1_init']
        ca2_init = mms.solution['c_a2_init']
        cb1_init = mms.solution['c_b1_init']
        cb2_init = mms.solution['c_b2_init']
        cc1_init = mms.solution['c_c1_init']
        cc2_init = mms.solution['c_c2_init']

        # get source terms from MMS equation for concentrations
        fca1 = mms.rhs['volume_c_a1']        # concentration a in domain 1
        fcb1 = mms.rhs['volume_c_b1']        # concentration b in domain 1
        fcc1 = mms.rhs['volume_c_c1']        # concentration a in domain 2
        fca2 = mms.rhs['volume_c_a2']        # concentration a in domain 2
        fcb2 = mms.rhs['volume_c_b2']        # concentration b in domain 2
        fcc2 = mms.rhs['volume_c_c2']        # concentration b in domain 2

        # coupling function on membrane interface for concentrations
        g_robin_a1 = mms.rhs['bdry']['u_a1']
        g_robin_b1 = mms.rhs['bdry']['u_b1']
        g_robin_c1 = mms.rhs['bdry']['u_c1']
        g_robin_a2 = mms.rhs['bdry']['u_a2']
        g_robin_b2 = mms.rhs['bdry']['u_b2']
        g_robin_c2 = mms.rhs['bdry']['u_c2']

        # Neumann condition on exterior boundary
        bdry_a = mms.rhs['bdry']['neumann_a']
        bdry_b = mms.rhs['bdry']['neumann_b']
        bdry_c = mms.rhs['bdry']['neumann_c']

        g_a = 1.0
        g_b = 1.0
        g_c = 1.0

        # diffusion coefficients for each sub-domain
        D_a_sub = {1:Constant(D_a1), 0:Constant(D_a2)}
        D_b_sub = {1:Constant(D_b1), 0:Constant(D_b2)}
        D_c_sub = {1:Constant(D_c1), 0:Constant(D_c2)}

        # coupling coefficients for each sub-domain
        C_a_sub = {1:Constant(C_a1), 0:Constant(C_a2)}
        C_b_sub = {1:Constant(C_b1), 0:Constant(C_b2)}
        C_c_sub = {1:Constant(C_c1), 0:Constant(C_c2)}

        # initial concentrations for each sub-domain
        a_init_sub = {1:ca1_init, 0:ca2_init}
        b_init_sub = {1:cb1_init, 0:cb2_init}
        c_init_sub = {1:cc1_init, 0:cc2_init}

        # create ions
        ion_a = {'D_sub':D_a_sub,
                 'z':z_a,
                 'c_init_sub': a_init_sub,
                 'f1': fca1, 'f2': fca2,
                 'g_robin_1': g_robin_a1, 'g_robin_2': g_robin_a2,
                 'bdry':bdry_a,
                 'C_sub': C_a_sub,
                 'name':'Na'}

        ion_b = {'D_sub':D_b_sub,
                 'z':z_b,
                 'c_init_sub': b_init_sub,
                 'f1': fcb1, 'f2': fcb2,
                 'g_robin_1': g_robin_b1, 'g_robin_2': g_robin_b2,
                 'bdry':bdry_b,
                 'C_sub': C_b_sub,
                 'name':'K'}

        ion_c = {'D_sub':D_c_sub,
                 'z':z_c,
                 'c_init_sub': c_init_sub,
                 'f1': fcc1, 'f2': fcc2,
                 'g_robin_1': g_robin_c1, 'g_robin_2': g_robin_c2,
                 'bdry':bdry_c,
                 'C_sub': C_c_sub,
                 'name':'Cl'}

        # the final ion will be eliminated
        ion_list = [ion_a, ion_b, ion_c]

        # Set membrane parameters
        membrane_params = namedtuple('membrane_params', ('g_a_leak',
                                     'g_b_leak', 'g_c_leak'))(g_a, g_b, g_c)

        S = Solver(params=params, ion_list=ion_list, degree_emi=degree,
                degree_knp=degree, mms=mms)

        S.setup_domain(mesh, subdomains, surfaces)      # setup domain
        S.setup_parameters()                            # setup physical parameters
        S.setup_FEM_spaces()                            # setup function spaces and numerical parameters

        direct_emi = True
        direct_knp = True

        # set parameters
        solver_params = namedtuple('solver_params', ('direct_emi',
            'direct_knp', 'resolution'))(direct_emi, direct_knp, resolution)

        #uh_ca, uh_cb, uh_phi = S.solve_system(Tstop, dt, t)
        uh, uh_cc = S.solve_system_passive(Tstop, t, solver_params, membrane_params)

        uh_ca = uh[0]; uh_cb = uh[1]; uh_phi = uh[2]

        # Compute L^2 error (on subdomains)
        dX = Measure('dx', domain=mesh, subdomain_data=subdomains)

        # compute error concentration a
        error_ca = inner(ca2 - uh_ca, ca2 - uh_ca)*dX(0,
                metadata={'quadrature_degree': 5}) + inner(ca1 - uh_ca, ca1 -
                        uh_ca)*dX(1, metadata={'quadrature_degree': 5})
        error_ca = sqrt(abs(assemble(error_ca)))

        # compute error concentration b
        error_cb = inner(cb2 - uh_cb, cb2 - uh_cb)*dX(0,
                metadata={'quadrature_degree': 5}) + inner(cb1 - uh_cb, cb1 -
                        uh_cb)*dX(1, metadata={'quadrature_degree': 5})
        error_cb = sqrt(abs(assemble(error_cb)))

        # compute error phi with norm for null_space solver for phi
        phi1_m_e = Constant(assemble(phi1*dX(1, metadata={'quadrature_degree': 5})))
        phi2_m_e = Constant(assemble(phi2*dX(0, metadata={'quadrature_degree': 5})))
        phi_mean_e = phi1_m_e + phi2_m_e

        phi1_m_a = Constant(assemble(uh_phi*dX(1, metadata={'quadrature_degree': 5})))
        phi2_m_a = Constant(assemble(uh_phi*dX(0, metadata={'quadrature_degree': 5})))
        phi_mean_a = phi1_m_a + phi2_m_a

        phi_mean = - phi_mean_a + phi_mean_e

        error_phi = inner(phi2 - phi_mean - uh_phi, phi2 - phi_mean - uh_phi)*dX(0, metadata={'quadrature_degree': 5}) \
                  + inner(phi1 - phi_mean - uh_phi, phi1 - phi_mean - uh_phi)*dX(1, metadata={'quadrature_degree': 5})
        #error_phi = inner(phi2  - uh_phi, phi2 - uh_phi)*dX(0, metadata={'quadrature_degree': 5}) \
                  #+ inner(phi1  - uh_phi, phi1 - uh_phi)*dX(1, metadata={'quadrature_degree': 5})

        error_phi = sqrt(abs(assemble(error_phi)))

        # append mesh size and errors
        hs.append(mesh.hmin())
        errors_ca.append(error_ca)
        errors_cb.append(error_cb)
        errors_phi.append(error_phi)

        if len(errors_ca) > 1:
            rate_ca = np.log(errors_ca[-1]/errors_ca[-2])/np.log(hs[-1]/hs[-2])
            rates_ca.append(rate_ca)
        else:
            rate_ca = np.nan

        if len(errors_cb) > 1:
            rate_cb = np.log(errors_cb[-1]/errors_cb[-2])/np.log(hs[-1]/hs[-2])
            rates_cb.append(rate_cb)
        else:
            rate_cb = np.nan

        if len(errors_phi) > 1:
            rate_phi = np.log(errors_phi[-1]/errors_phi[-2])/np.log(hs[-1]/hs[-2])
            rates_phi.append(rate_phi)
        else:
            rate_phi = np.nan

        msg = f'|ca-cah|_0 = {error_ca:.4E} [{rate_ca:.2f}]'
        mesh.mpi_comm().rank == 0 and print(GREEN % msg)
        msg = f'|cb-cbh|_0 = {error_cb:.4E} [{rate_cb:.2f}]'
        mesh.mpi_comm().rank == 0 and print(GREEN % msg)
        msg = f'|phi-phih|_0 = {error_phi:.4E} [{rate_phi:.2f}]'
        mesh.mpi_comm().rank == 0 and print(GREEN % msg)

        i += 1

    print("concentration a")
    print(rates_ca)
    for i in range(len(hs)):
        print(hs[i], errors_ca[i])

    print("concentration b")
    print(rates_cb)
    for i in range(len(hs)):
        print(hs[i], errors_cb[i])

    print("phi")
    print(rates_phi)
    for i in range(len(hs)):
        print(hs[i], errors_phi[i])
