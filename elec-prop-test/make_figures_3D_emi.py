import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from fenics import * 
import string

from knpemidg import pcws_constant_project
from knpemidg import interface_normal, plus, minus

JUMP = lambda f, n: minus(f, n) - plus(f, n)

# set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

path = 'results/data/'

c_1 = '#%02x%02x%02x' % (255, 0, 255)  # pink
c_2 = '#%02x%02x%02x' % (54, 92, 141)  # blue
c_3 = '#%02x%02x%02x' % (255, 165, 0)  # orange
phi_M_c = '#%02x%02x%02x' % (54, 92, 141)

def get_time_series(dt, T, fname, x, y, z):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e4
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement(2*[P1]))
    V = FunctionSpace(mesh, P1)

    u = Function(W)
    v = Function(V)
    w = Function(V)

    f_Na = Function(V)
    f_K = Function(V)
    f_Cl = Function(V)
    f_phi = Function(V)

    Na = []
    K = []
    Cl = []
    phi = []

    for n in range(1, int(T/dt)):

            # read file
            hdf5file.read(u, "/concentrations/vector_" + str(n))

            # K concentrations
            assign(f_K, u.sub(0))
            K.append(f_K(x, y, z))

            # Cl concentrations
            assign(f_Cl, u.sub(1))
            Cl.append(f_Cl(x, y, z))

            # Na concentrations
            hdf5file.read(v, "/elim_concentration/vector_" + str(n))
            assign(f_Na, v)
            Na.append(f_Na(x, y, z))

            # potential
            hdf5file.read(w, "/potential/vector_" + str(n))
            assign(f_phi, w)
            phi.append(f_phi(x, y, z))

    return Na, K, Cl, phi

def get_time_series_membrane(dt, T, fname, x_, y_, z_):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e4
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    x_min = 100.3; x_max = 100.4
    y_min = 0.76; y_max = 0.77
    z_min = 0.79; z_max = 0.81

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]

        point_1 = x_min < x[0] < x_max and \
                  y_min < x[1] < y_max and \
                  z_min < x[2] < z_max

        if point_1:
            surfaces[facet] = 10

    surfacesfile = File('surfaces_plot.pvd')
    surfacesfile << surfaces

    # define function space of piecewise constants on interface gamma for solution to ODEs
    Q = FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    phi_M = Function(Q)
    E_Na = Function(Q)
    E_K = Function(Q)

    # interface normal
    n_g = interface_normal(subdomains, mesh)

    dS = Measure('dS', domain=mesh, subdomain_data=surfaces)
    iface_size = assemble(Constant(1)*dS(10))

    P1 = FiniteElement('DG', mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement(2*[P1]))
    V = FunctionSpace(mesh, P1)

    v = Function(W)
    w_Na = Function(V)
    w_phi = Function(V)

    f_phi = Function(V)
    f_Na = Function(V)
    f_K = Function(V)

    phi_M_s = []
    E_Na_s = []
    E_K_s = []

    z_Na = 1; z_K = 1; temperature = 300; F = 96.485; R = 8.314

    for n in range(1, int(T/dt)):

            # potential
            hdf5file.read(w_phi, "/potential/vector_" + str(n))
            assign(f_phi, w_phi)

            # K concentrations
            hdf5file.read(v, "/concentrations/vector_" + str(n))
            assign(f_K, v.sub(0))
            E = R * temperature / (F * z_K) * ln(plus(f_K, n_g) / minus(f_K, n_g))
            assign(E_K, pcws_constant_project(E, Q))
            E_K_ = assemble(1.0/iface_size*avg(E_K)*dS(10))
            E_K_s.append(E_K_)

            # Na concentrations
            hdf5file.read(w_Na, "/elim_concentration/vector_" + str(n))
            assign(f_Na, w_Na)
            E = R * temperature / (F * z_Na) * ln(plus(f_Na, n_g) / minus(f_Na, n_g))
            assign(E_Na, pcws_constant_project(E, Q))
            E_Na_ = assemble(1.0/iface_size*avg(E_Na)*dS(10))
            E_Na_s.append(E_Na_)

            # update membrane potential
            phi_M_step = JUMP(f_phi, n_g)
            assign(phi_M, pcws_constant_project(phi_M_step, Q))
            phi_M_s.append(assemble(1.0/iface_size*avg(phi_M)*dS(10)))

    return phi_M_s, E_Na_s, E_K_s


def plot_3D_concentration(res, T, dt):

    time = np.arange(0, T-dt, dt)

    # at membrane of axon A (gamma)
    x_M_A = 100; y_M_A = 0.333; z_M_A = 0.8
    # 0.05 um above axon A (ECS)
    x_e_A = 100; y_e_A = 0.9; z_e_A = 0.9
    # mid point inside axon A (ICS)
    x_i_A = 100; y_i_A = 0.3; z_i_A = 0.6

    #################################################################
    # get data axon A is stimulated
    fname = 'results/data/3D_emi/results.h5'

    # trace concentrations
    phi_M, E_Na, E_K = get_time_series_membrane(dt, T, fname, x_M_A, y_M_A, z_M_A)

    # bulk concentrations
    Na_e, K_e, Cl_e, _ = get_time_series(dt, T, fname, x_e_A, y_e_A, z_e_A)
    Na_i, K_i, Cl_i, _ = get_time_series(dt, T, fname, x_i_A, y_i_A, z_i_A)

    #################################################################
    # get data axons BC are stimulated

    # Concentration plots
    fig = plt.figure(figsize=(12*0.9,12*0.9))
    ax = plt.gca()

    print("-------------------------")
    print("Na_e", Na_e[-1])
    print("K_e", K_e[-1])
    print("Cl_e", Cl_e[-1])

    print("Na_i", Na_i[-1])
    print("K_i", K_i[-1])
    print("Cl_i", Cl_i[-1])

    print("phi_M", phi_M[-1])
    print("E_Na", E_Na[-1])
    print("E_K", E_K[-1])
    print("-------------------------")

    # global kappa
    Na_i_init = Na_i[-1]
    Na_e_init = Na_e[-1]
    K_i_init = K_i[-1]
    K_e_init = K_e[-1]
    Cl_i_init = Cl_i[-1]
    Cl_e_init = Cl_e[-1]

    # Physical parameters (PDEs)
    C_M = 1.0                      # capacitance
    temperature = 300e3            # temperature (m K)
    R = 8.314e3                    # Gas Constant (m J/(K mol))
    F = 96485e3                    # Faraday's constant (mC/ mol)

    D_Na = 1.33e-8       # diffusion coefficients Na (cm/ms)
    D_K = 1.96e-8        # diffusion coefficients K (cm/ms)
    D_Cl = 2.03e-8       # diffusion coefficients Cl (cm/ms)

    psi = F/(R * temperature)

    kappa_i = F * psi * (float(D_Na) * float(Na_i_init) \
                      + float(D_K) * float(K_i_init) \
                      + float(D_Cl) * float(Cl_i_init))

    kappa_e = F * psi * (float(D_Na) * float(Na_e_init) \
                       + float(D_K) * float(K_e_init) \
                       + float(D_Cl) * float(Cl_e_init))

    E_K = float((R * temperature)/F)

    print("kappa_i", kappa_i)
    print("kappa_e", kappa_e)

    print(kappa_e)
    print(E_K)
    print(float((F*F)/(R*temperature)))

    ax1 = fig.add_subplot(3,3,1)
    plt.title(r'Na$^+$ concentration (ECS)')
    plt.ylabel(r'[Na]$_e$ (mM)')
    plt.plot(Na_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,2)
    plt.title(r'K$^+$ concentration (ECS)')
    plt.ylabel(r'[K]$_e$ (mM)')
    plt.plot(K_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,3)
    plt.title(r'Cl$^-$ concentration (ECS)')
    plt.ylabel(r'[Cl]$_e$ (mM)')
    plt.plot(Cl_e, linewidth=3, color='b')

    ax2 = fig.add_subplot(3,3,4)
    plt.title(r'Na$^+$ concentration (ICS)')
    plt.ylabel(r'[Na]$_i$ (mM)')
    plt.plot(Na_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,5)
    plt.title(r'K$^+$ concentration (ICS)')
    plt.ylabel(r'[K]$_i$ (mM)')
    plt.plot(K_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,6)
    plt.title(r'Cl$^-$ concentration (ICS)')
    plt.ylabel(r'[Cl]$_i$ (mM)')
    plt.plot(Cl_i,linewidth=3, color='r')

    ax5 = fig.add_subplot(3,3,7)
    plt.title(r'Membrane potential')
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(phi_M, linewidth=3)

    ax6 = fig.add_subplot(3,3,8)
    plt.title(r'Na$^+$ reversal potential')
    plt.ylabel(r'E$_Na$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(E_K, linewidth=3)
    plt.plot(E_Na, linewidth=3)

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # save figure to file
    plt.savefig('results/figures/pot_con_3D_emi.svg', format='svg')

    f_phi_M = open('results/data/3D_emi/solver/phi_M_3D.txt', "w")
    for p in phi_M:
        f_phi_M.write("%.10f \n" % p)
    f_phi_M.close()

    return

def get_plottable_gamma_function(fname, n):
    """ Return list of values in given point (x, y, z) over time """

    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    # create mesh
    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    surfaces = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    hdf5file.read(mesh, "/mesh", False)
    mesh.coordinates()[:] *= 1e4
    hdf5file.read(subdomains, "/subdomains")
    hdf5file.read(surfaces, "/surfaces")

    mesh = mesh

    # create function spaces
    P1 = FiniteElement("DG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)

    # create functions
    f_phi = Function(V)
    w = Function(V)

    # potential
    hdf5file.read(w, "/potential/vector_" + str(n))
    assign(f_phi, w)

    # membrane potential
    Q = FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    dS = Measure('dS', domain=mesh, subdomain_data=surfaces)
    n_g = interface_normal(subdomains, mesh)
    phi_M = Function(Q)

    # update membrane potential
    phi_M_step = JUMP(f_phi, n_g)
    assign(phi_M, pcws_constant_project(phi_M_step, Q))

    phi_M_s = []
    x_s = []

    x_min = 5.0; x_max = 205.0
    y_min = 0.76; y_max = 0.77
    z_min = 0.79; z_max = 0.81

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        line = (x_min <= x[0] <= x_max and y_min <= x[1] <= y_max and  z_min <= x[2] <= z_max)
        if (surfaces[facet] == 1 and line):
            print(x[0], x[1], x[2])
            surfaces[facet] = 10

            iface_size = assemble(Constant(1)*dS(10))
            phi_M_s.append(assemble(1.0/iface_size*avg(phi_M)*dS(10)))
            x_s.append(x[0])

        #if (surfaces[facet] == 1):
            #print(x[0], x[1], x[2])

    return phi_M_s, x_s

def plot_membrane_space(fname):

    plt.figure(figsize=(4.6, 3.3))
    plt.xlabel(r"x-position ($\mu$m)")
    plt.ylabel("$\phi_M$ (mV)")

    header = "x r0 r1 r2"
    phi_dat = []

    for idx, n in enumerate([1, 5, 10, 15, 20, 25, 30, 35, 99]):
        phi_M_n, x_s = get_plottable_gamma_function(fname, n)
        n_new = n/10
        x = np.linspace(5, 200, len(phi_M_n))
        plt.plot(x_s, phi_M_n, linewidth=2.5, label=r"%d ms" % n_new)
        phi_dat.append(phi_M_n)

    plt.legend()
    plt.tight_layout()
    # save figure
    plt.savefig("results/figures/phi_M_space_emi.svg")
    plt.close()

    #phi_dat.insert(0, x)
    #a_phi = np.asarray(phi_dat)
    #np.savetxt("results/figures/phi_space_emi.dat", np.transpose(a_phi), delimiter=" ", header=header)

    return

def get_velocity(fname, T, dt):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e4
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        # mark top point
        if surfaces[facet] == 1 and (20 < x[0] < 20.5) and (x[1] > 0.79) and (x[2] > 0.76):
            print("point 1", x[0], x[1], x[2])
            surfaces[facet] = 10
        # mark bottom point
        if surfaces[facet] == 1 and (120 < x[0] < 120.5) and (x[1] > 0.79) and (x[2] > 0.76):
            print("point 2", x[0], x[1], x[2])
            surfaces[facet] = 20

    # define function space of piecewise constants on interface gamma for solution to ODEs
    Q = FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    phi_M = Function(Q)

    # interface normal
    n_g = interface_normal(subdomains, mesh)

    dS = Measure('dS', domain=mesh, subdomain_data=surfaces)
    iface_size_10 = assemble(Constant(1)*dS(10))
    iface_size_20 = assemble(Constant(1)*dS(20))

    P1 = FiniteElement('DG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)

    w_phi = Function(V)
    f_phi = Function(V)

    phi_M_s_1 = []
    phi_M_s_2 = []

    time_point_1 = 0
    time_point_2 = 0

    for n in range(1, int(T/dt)):

            # potential
            hdf5file.read(w_phi, "/potential/vector_" + str(n))
            assign(f_phi, w_phi)

            # update membrane potential
            phi_M_step = JUMP(f_phi, n_g)
            assign(phi_M, pcws_constant_project(phi_M_step, Q))

            if 1.0e3*assemble(1.0/iface_size_10*avg(phi_M)*dS(10)) > 20 and time_point_1 == 0:
                time_point_1 = n*0.1

            if 1.0e3*assemble(1.0/iface_size_10*avg(phi_M)*dS(20)) > 20 and time_point_2 == 0:
                time_point_2 = n*0.1

            print("t1", time_point_1)
            print("t2", time_point_2)

            # if membrane potential has reach 0 in both points then break
            if (time_point_1 > 0) and (time_point_2 > 0):
                break

    delta_t = (time_point_2 - time_point_1) # ms
    delta_x = 100 # um

    print("velocity (um/ms)", delta_x/delta_t)

    return phi_M_s_1, phi_M_s_2

# create directory for figures
if not os.path.isdir('results/figures'):
    os.mkdir('results/figures')

# create figures
res_3D = '0' # mesh resolution for 3D axon bundle
T = 10
dt = 0.1

fname = 'results/data/3D_emi/results.h5'

plot_3D_concentration(res_3D, T, dt)
plot_membrane_space(fname)
get_velocity(fname, T, dt)
