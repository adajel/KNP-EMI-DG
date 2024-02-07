from dolfin import *
import ulfy  # https://github.com/MiroK/ulfy
from collections import namedtuple
import sympy as sp

MMSData = namedtuple('MMSData', ('solution', 'rhs', 'normals'))

def setup_mms(params, t):
    '''We solve EMI on

    [       ]
    [  [ ]  ]
    [       ]

    domain
    '''
    order = 2
    mesh = UnitSquareMesh(2**(2+order), 2**(2+order), 'crossed')
    x, y = SpatialCoordinate(mesh)

    # We will vary this outside
    D_a1, D_a2, D_b1, D_b2, D_c1, D_c2,\
    C_a1, C_a2, C_b1, C_b2, C_c1, C_c2, C_phi, \
    z_a, z_b, z_c, dt, F, R, temperature = \
    Constant(1), Constant(1), Constant(1), Constant(1), \
    Constant(1), Constant(1), Constant(1), Constant(1), Constant(1), \
    Constant(1), Constant(1), Constant(1), Constant(1), Constant(1), \
    Constant(1), \
    Constant(1), Constant(1), Constant(1), Constant(1), Constant(1)

    # define exact solutions
    k_a1 = 1 + (x + y) + 0.3 * cos(2 * pi * t)
    k_b1 = 1 + (x + y) + 0.3 * cos(2 * pi * t)
    k_c1 = - 1/z_c * (z_a*k_a1 + z_b*k_b1)
    phi_1 = (1 + x + y) * (1 + t**2)

    k_a2 = 1 + (x + y) + 0.5 * sin(2 * pi * t)
    k_b2 = 1 + (x + y) + 0.5 * sin(2 * pi * t)
    k_c2 = - 1/z_c * (z_a*k_a2 + z_b*k_b2)
    phi_2 = (1 + x - y) * (1 + t**2)

    k_a1_dt = - 0.3 * 2 * pi * sin(2 * pi * t)
    k_b1_dt = - 0.3 * 2 * pi * sin(2 * pi * t)
    k_c1_dt = 0
    k_a2_dt = 0.5 * 2 * pi * cos(2 * pi * t)
    k_b2_dt = 0.5 * 2 * pi * cos(2 * pi * t)
    k_c2_dt = 0

    psi = F/(R*temperature)

    # shorthand - linearised ion fluxes
    J_a1 = - D_a1 * grad(k_a1) - z_a * D_a1 * psi * k_a1 * grad(phi_1)
    J_b1 = - D_b1 * grad(k_b1) - z_b * D_b1 * psi * k_b1 * grad(phi_1)
    J_c1 = - D_c1 * grad(k_c1) - z_c * D_c1 * psi * k_c1 * grad(phi_1)
    J_a2 = - D_a2 * grad(k_a2) - z_a * D_a2 * psi * k_a2 * grad(phi_2)
    J_b2 = - D_b2 * grad(k_b2) - z_b * D_b2 * psi * k_b2 * grad(phi_2)
    J_c2 = - D_c2 * grad(k_c2) - z_c * D_c2 * psi * k_c2 * grad(phi_2)

    # calculate source terms
    f_k_a1 = k_a1_dt + div(J_a1)
    f_k_b1 = k_b1_dt + div(J_b1)
    f_k_c1 = k_c1_dt + div(J_c1)
    f_phi_1 = F*(z_a*div(J_a1) + z_b*div(J_b1) + + z_c*div(J_c1))

    f_k_a2 = k_a2_dt + div(J_a2)
    f_k_b2 = k_b2_dt + div(J_b2)
    f_k_c2 = k_c2_dt + div(J_c2)
    f_phi_2 = F*(z_a*div(J_a2) + z_b*div(J_b2) + z_c*div(J_c2))

    # Normal will point from inner to outer; from 1 to 2
    normals = list(map(Constant, ((-1, 0), (0, -1), (1, 0), (0, 1))))

    # We have that g_k_a1 = phi_i - phi_e - 1/C_a1 * J_a1 \cdot n1
    g_k_a1 = tuple(
              phi_1 - phi_2 - (1/C_a1) * dot(J_a1, n1)
              for n1 in normals
    )

    # We have that g_k_b1 = phi_i - phi_e - 1/C_b1 * J_b1 \cdot n1
    g_k_b1 = tuple(
              phi_1 - phi_2 - (1/C_b1) * dot(J_b1, n1)
              for n1 in normals
    )

    # We have that g_k_b1 = phi_i - phi_e - 1/C_b1 * J_b1 \cdot n1
    g_k_c1 = tuple(
              phi_1 - phi_2 - (1/C_c1) * dot(J_c1, n1)
              for n1 in normals
    )

    # We have that g_k_a2 = phi_i - phi_e - 1/C_a2 * J_a2 \cdot n1
    g_k_a2 = tuple(
              phi_1 - phi_2 - (1/C_a2) * dot(J_a2, n1)
              for n1 in normals
    )

    # We have that g_k_b2 = phi_i - phi_e - 1/C_b2 * J_b2 \cdot n1
    g_k_b2 = tuple(
              phi_1 - phi_2 - (1/C_b2) * dot(J_b2, n1)
              for n1 in normals
    )

    # We have that g_k_b2 = phi_i - phi_e - 1/C_b2 * J_b2 \cdot n1
    g_k_c2 = tuple(
              phi_1 - phi_2 - (1/C_c2) * dot(J_c2, n1)
              for n1 in normals
    )

    # We have that f = phi_i - phi_e - dt/C_M * I_M, where I_M = F sum_c dot(z_c*J_c1, n1)
    g_phi = tuple(
              phi_1 - phi_2 - (1/C_phi)*F*(dot(z_a*J_a1, n1) + \
                      dot(z_b*J_b1, n1) + dot(z_c*J_c1, n1))
              for n1 in normals
    )

    # We don't have 0 on the rhs side of the interface, we have
    # F sum_k (z_k_i*J_k_i \cdot n_i) + F sum_k (z_k_e*J_k_e \cdot n_e) = g_J_phi
    g_J_phi = tuple(
       - F * z_a * (dot(J_a1, n1) - dot(J_a2, n1)) \
       - F * z_b * (dot(J_b1, n1) - dot(J_b2, n1)) \
       - F * z_c * (dot(J_c1, n1) - dot(J_c2, n1)) \
         for n1 in normals
    )

    # What we want to substitute
    D_a1_, D_a2_, D_b1_, D_b2_, D_c1_, D_c2_, \
    C_a1_, C_a2_, C_b1_, C_b2_, C_c1_, C_c2_, C_phi_, \
    z_a_, z_b_, z_c_, dt_, F_, R_, temperature_, t_ = \
            sp.symbols('D_a1, D_a2, D_b1, D_b2, D_c1, D_c2,\
                        C_a1, C_a2, C_b1, C_b2, C_c1, C_c2, C_phi, \
                       z_a, z_b, z_c, dt, F, R, temperature, t')

    subs = {D_a1:D_a1_, D_a2:D_a2_, D_b1:D_b1_, D_b2:D_b2_, D_c1:D_c1_, D_c2:D_c2_,\
            C_a1:C_a1_, C_a2:C_a2_, C_b1:C_b1_, C_b2:C_b2_, C_c1:C_c1_, C_c2:C_c2_,\
            C_phi:C_phi_, \
            z_a:z_a_, z_b:z_b_, z_c:z_c_, dt:dt_, F:F_, t:t_}

    as_expression = lambda f, subs=subs: ulfy.Expression(f, subs=subs, degree=4,
                                                         D_a1=params.D_a1,
                                                         D_a2=params.D_a2,
                                                         D_b1=params.D_b1,
                                                         D_b2=params.D_b2,
                                                         D_c1=params.D_c1,
                                                         D_c2=params.D_c2,
                                                         C_a1=params.C_a1,
                                                         C_a2=params.C_a2,
                                                         C_b1=params.C_b1,
                                                         C_b2=params.C_b2,
                                                         C_c1=params.C_c1,
                                                         C_c2=params.C_c2,
                                                         C_phi=params.C_phi,
                                                         z_a=params.z_a,
                                                         z_b=params.z_b,
                                                         z_c=params.z_c,
                                                         F=params.F,
                                                         R=params.R,
                                                         temperature=params.temperature,
                                                         dt=params.dt,
                                                         t=t)

    k_a1_exact, k_b1_exact, k_c1_exact, phi_1_exact, J_a1_exact, J_b1_exact, J_c1_exact = \
        map(as_expression, (k_a1, k_b1, k_c1, phi_1, J_a1, J_b1, J_c1))

    k_a2_exact, k_b2_exact, k_c2_exact, phi_2_exact, J_a2_exact, J_b2_exact, J_c2_exact = \
        map(as_expression, (k_a2, k_b2, k_c2, phi_2, J_a2, J_b2, J_c2))

    return MMSData(solution={'c_a1': k_a1_exact,
                             'c_b1': k_b1_exact,
                             'c_c1': k_c1_exact,
                             'phi_1': phi_1_exact,
                             'c_a2': k_a2_exact,
                             'c_b2': k_b2_exact, \
                             'c_c2': k_c2_exact, \
                             'phi_2': phi_2_exact},
                   rhs={'volume_c_a1': as_expression(f_k_a1),
                        'volume_c_b1': as_expression(f_k_b1),
                        'volume_c_c1': as_expression(f_k_c1),
                        'volume_phi_1': as_expression(f_phi_1),
                        'volume_c_a2': as_expression(f_k_a2),
                        'volume_c_b2': as_expression(f_k_b2),
                        'volume_c_c2': as_expression(f_k_c2),
                        'volume_phi_2': as_expression(f_phi_2),
                        'bdry': {'neumann_a': J_a2_exact,
                                 'neumann_b': J_b2_exact,
                                 'neumann_c': J_c2_exact,
                                 'stress': dict(enumerate(map(as_expression, g_J_phi), 1)),
                                 'u_phi': dict(enumerate(map(as_expression, g_phi), 1)),
                                 'u_a1': dict(enumerate(map(as_expression, g_k_a1), 1)),
                                 'u_b1': dict(enumerate(map(as_expression, g_k_b1), 1)),
                                 'u_c1': dict(enumerate(map(as_expression, g_k_c1), 1)),
                                 'u_a2': dict(enumerate(map(as_expression, g_k_a2), 1)),
                                 'u_b2': dict(enumerate(map(as_expression, g_k_b2), 1)),
                                 'u_c2': dict(enumerate(map(as_expression, g_k_c2), 1))}},
                   normals=[])
