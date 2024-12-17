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

def get_time_series(dt, T, fname, x_e, y_e, x_i, y_i):
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

    Na_e = []
    K_e = []
    Cl_e = []
    phi_e = []

    Na_i = []
    K_i = []
    Cl_i = []
    phi_i = []

    for n in range(1, int(T/dt)):
            print(n)

            # read file
            hdf5file.read(u, "/concentrations/vector_" + str(n))

            # K concentrations
            assign(f_K, u.sub(0))
            K_e.append(f_K(x_e, y_e))
            K_i.append(f_K(x_i, y_i))

            # Cl concentrations
            assign(f_Cl, u.sub(1))
            Cl_e.append(f_Cl(x_e, y_e))
            Cl_i.append(f_Cl(x_i, y_i))

            # Na concentrations
            hdf5file.read(v, "/elim_concentration/vector_" + str(n))
            assign(f_Na, v)
            Na_e.append(f_Na(x_e, y_e))
            Na_i.append(f_Na(x_i, y_i))

            # potential
            hdf5file.read(w, "/potential/vector_" + str(n))
            assign(f_phi, w)
            phi_e.append(1.0e3*f_phi(x_e, y_e))
            phi_i.append(1.0e3*f_phi(x_i, y_i))

    return Na_e, K_e, Cl_e, phi_e, Na_i, K_i, Cl_i, phi_i

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

    x_min = x_ - 0.5
    x_max = x_ + 0.5
    y_min = y_ - 0.5
    y_max = y_ + 0.5

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        point_1 = (y_min <= x[1] <= y_max and x_min <= x[0] <= x_max)

        if point_1 and surfaces[facet] == 1:
            print(x[0], x[1])
            surfaces[facet] = 10
            break

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
            print(n)

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


def plot_2D_concentration(dt, T):

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
    Na_e, K_e, Cl_e, _, Na_i, K_i, Cl_i, _ = get_time_series(dt, T, fname, \
            x_e_A, y_e_A, x_i_A, y_i_A)

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
        f_phi_M.write("%.10f \n" % p)
    f_phi_M.close()

    f_K_e = open('results/data/2D/solver/K_ECS_2D.txt', "w")
    for p in K_e:
        f_K_e.write("%.10f \n" % p)
    f_K_e.close()

    f_K_i = open('results/data/2D/solver/K_ICS_2D.txt', "w")
    for p in K_i:
        f_K_i.write("%.10f \n" % p)
    f_K_i.close()

    f_Na_e = open('results/data/2D/solver/Na_ECS_2D.txt', "w")
    for p in Na_e:
        f_Na_e.write("%.10f \n" % p)
    f_Na_e.close()

    f_Na_i = open('results/data/2D/solver/Na_ICS_2D.txt', "w")
    for p in Na_i:
        f_Na_i.write("%.10f \n" % p)
    f_Na_i.close()

    f_E_Na = open('results/data/2D/solver/E_Na_2D.txt', "w")
    for p in E_Na:
        f_E_Na.write("%.10f \n" % p)
    f_E_Na.close()

    f_E_K = open('results/data/2D/solver/E_K_2D.txt', "w")
    for p in E_K:
        f_E_K.write("%.10f \n" % p)
    f_E_K.close()

    return

# create directory for figures
if not os.path.isdir('results/figures'):
    os.mkdir('results/figures')

# create figures
dt = 1.0e-4
T = 3.0e-1

plot_2D_concentration(dt, T)
