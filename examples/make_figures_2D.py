import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from fenics import * 
import string

from knpemidg.utils import pcws_constant_project
from knpemidg.utils import interface_normal, plus, minus

JUMP = lambda f, n: minus(f, n) - plus(f, n)

# set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

path = 'results/data/'

def get_time_series(dt, T, fname, x, y):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
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
            K.append(f_K(x, y))

            # Cl concentrations
            assign(f_Cl, u.sub(1))
            Cl.append(f_Cl(x, y))

            # Na concentrations
            hdf5file.read(v, "/elim_concentration/vector_" + str(n))
            assign(f_Na, v)
            Na.append(f_Na(x, y))

            # potential
            hdf5file.read(w, "/potential/vector_" + str(n))
            assign(f_phi, w)
            phi.append(1.0e3*f_phi(x, y))

    return Na, K, Cl, phi

def get_time_series_membrane(dt, T, fname, x_, y_):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    x_min = 24.15
    x_max = 24.2
    y_min = 0.9
    y_max = 1.1

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        point_1 = (y_min <= x[1] <= y_max and x_min <= x[0] <= x_max)
        if point_1 and surfaces[facet] == 1:
            print(x[0], x[1])
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

    z_Na = 1; z_K = 1; temperature = 300; F = 96485; R = 8.314

    for n in range(1, int(T/dt)):

            # potential
            hdf5file.read(w_phi, "/potential/vector_" + str(n))
            assign(f_phi, w_phi)

            # K concentrations
            hdf5file.read(v, "/concentrations/vector_" + str(n))
            assign(f_K, v.sub(0))
            E = R * temperature / (F * z_K) * ln(plus(f_K, n_g) / minus(f_K, n_g))
            assign(E_K, pcws_constant_project(E, Q))
            E_K_ = 1.0e3*assemble(1.0/iface_size*avg(E_K)*dS(10))
            E_K_s.append(E_K_)

            # Na concentrations
            hdf5file.read(w_Na, "/elim_concentration/vector_" + str(n))
            assign(f_Na, w_Na)
            E = R * temperature / (F * z_Na) * ln(plus(f_Na, n_g) / minus(f_Na, n_g))
            assign(E_Na, pcws_constant_project(E, Q))
            E_Na_ = 1.0e3*assemble(1.0/iface_size*avg(E_Na)*dS(10))
            E_Na_s.append(E_Na_)

            # update membrane potential
            phi_M_step = JUMP(f_phi, n_g)
            assign(phi_M, pcws_constant_project(phi_M_step, Q))
            phi_M_s.append(1.0e3*assemble(1.0/iface_size*avg(phi_M)*dS(10)))

    return phi_M_s, E_Na_s, E_K_s

def get_time_series_gating(dt, T, fname, x_, y_):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    x_min = x_; x_max = x_ + 0.08
    y_min = y_ - 0.01; y_max = y_

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        point_1 = (y_min <= x[1] <= y_max and x_min <= x[0] <= x_max)
        if point_1:
            print(x[0], x[1])
            surfaces[facet] = 10

    surfacesfile = File('surfaces_plot.pvd')
    surfacesfile << surfaces

    # define function space of piecewise constants on interface gamma for solution to ODEs
    Q = FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)

    v_n_HH = Function(Q)
    v_m_HH = Function(Q)
    v_h_HH = Function(Q)

    f_n_HH = Function(Q)
    f_m_HH = Function(Q)
    f_h_HH = Function(Q)

    # interface normal
    n_g = interface_normal(subdomains, mesh)

    dS = Measure('dS', domain=mesh, subdomain_data=surfaces)
    iface_size = assemble(Constant(1)*dS(10))

    P1 = FiniteElement('DG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)

    n_HH_s = []
    m_HH_s = []
    h_HH_s = []

    for n in range(1, int(T/dt)):

            # gating
            hdf5file.read(v_n_HH, "/n_HH/vector_" + str(n))
            assign(f_n_HH, v_n_HH)

            hdf5file.read(v_m_HH, "/m_HH/vector_" + str(n))
            assign(f_m_HH, v_m_HH)

            hdf5file.read(v_h_HH, "/h_HH/vector_" + str(n))
            assign(f_h_HH, v_h_HH)

            # n
            n_HH_ = assemble(1.0/iface_size*avg(f_n_HH)*dS(10))
            n_HH_s.append(n_HH_)

            # m
            m_HH_ = assemble(1.0/iface_size*avg(f_m_HH)*dS(10))
            m_HH_s.append(m_HH_)

            # h
            h_HH_ = assemble(1.0/iface_size*avg(f_h_HH)*dS(10))
            h_HH_s.append(h_HH_)

    return n_HH_s, m_HH_s, h_HH_s


def plot_2D_concentration(res):

    dt = 1.0e-4
    T = 1.0e-1

    temperature = 300 # temperature (K)
    F = 96485         # Faraday's constant (C/mol)
    R = 8.314         # Gas constant (J/(K*mol))

    time = 1.0e3*np.arange(0, T-dt, dt)

    # at membrane of axon A (gamma)
    x_M_A = 25; y_M_A = 3
    # 0.05 um above axon A (ECS)
    x_e_A = 25; y_e_A = 3.5
    # mid point inside axon A (ICS)
    x_i_A = 25; y_i_A = 2

    #################################################################
    # get data axon A is stimulated
    fname = 'results/data/2D/results.h5'

    # trace concentrations
    phi_M, E_Na, E_K = get_time_series_membrane(dt, T, fname, x_M_A, y_M_A)

    # bulk concentrations
    Na_e, K_e, Cl_e, _ = get_time_series(dt, T, fname, x_e_A, y_e_A)
    Na_i, K_i, Cl_i, _ = get_time_series(dt, T, fname, x_i_A, y_i_A)

    #################################################################
    # get data axons BC are stimulated

    # Concentration plots
    fig = plt.figure(figsize=(12*0.9,12*0.9))
    ax = plt.gca()

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

    ax6 = fig.add_subplot(3,3,9)
    plt.legend()

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # save figure to file
    plt.savefig('results/figures/pot_con_2D.svg', format='svg')

    f_phi_M = open('results/data/2D/solver/phi_M_2D.txt', "w")
    for p in phi_M:
        f_phi_M.write("%.10f \n" % p*1000)
    f_phi_M.close()

    return

# create directory for figures
if not os.path.isdir('results/figures'):
    os.mkdir('results/figures')

# create figures
res_2D = '2'
plot_2D_concentration(res_2D)
