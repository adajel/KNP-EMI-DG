import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from fenics import * 
import string

from surf_plot.vtk_io import DltWriter
from surf_plot.dlt_embedding import P0surf_to_DLT0_map

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

def write_to_pvd(dt, T, fname):
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

    Na = Function(V)
    K = Function(V)
    Cl = Function(V)
    phi = Function(V)

    f_phi = File('results/pvd/pot.pvd')
    f_K = File('results/pvd/K.pvd')
    f_Na = File('results/pvd/Na.pvd')
    f_Cl = File('results/pvd/Cl.pvd')

    for n in range(1, int(T/dt)):

        # read file
        hdf5file.read(u, "/concentrations/vector_" + str(n))

        # K concentrations
        assign(K, u.sub(0))
        # Cl concentrations
        assign(Cl, u.sub(1))

        # Na concentrations
        hdf5file.read(v, "/elim_concentration/vector_" + str(n))
        assign(Na, v)

        # potential
        hdf5file.read(w, "/potential/vector_" + str(n))
        assign(phi, w)

        f_Na << Na, n
        f_K << K, n
        f_Cl << Cl, n
        f_phi << phi, n

    return


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

def get_time_series_membrane(dt, T, fname, x, y, z):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e4
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    x_min = x - 0.5; x_max = x + 0.1
    y_min = y - 0.5; y_max = y + 0.1
    z_min = z; z_max = z + 0.04

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        point_1 = y_min <= x[1] <= y_max \
              and x_min <= x[0] <= x_max \
              and z_min <= x[2] <= z_max
        if point_1 and (surfaces[facet] == 1 or surfaces[facet] == 2):
            print(x[0], x[1], x[2])
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

    w_phi_ODE = Function(Q)
    f_phi_ODE = Function(Q)

    f_phi = Function(V)
    f_Na = Function(V)
    f_K = Function(V)

    phi_M_s = []
    #phi_M_ODE_s = []
    E_Na_s = []
    E_K_s = []

    z_Na = 1; z_K = 1; temperature = 300e3; F = 96485e3; R = 8.314e3

    for n in range(1, int(T/dt)):

            # update membrane potential
            hdf5file.read(w_phi, "/potential/vector_" + str(n))
            assign(f_phi, w_phi)
            phi_M_step = JUMP(f_phi, n_g)
            assign(phi_M, pcws_constant_project(phi_M_step, Q))
            phi_M_s.append(assemble(1.0/iface_size*avg(phi_M)*dS(10)))

            #hdf5file.read(w_phi_ODE, "/membranepotential/vector_" + str(n))
            #assign(f_phi_ODE, w_phi_ODE)
            #phi_M_ODE_s.append(assemble(1.0/iface_size*avg(f_phi_ODE)*dS(10)))

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

    return phi_M_s, phi_M_s, E_Na_s, E_K_s

def plot_3D_concentration_neuron(res, T, dt, fname):

    time = np.arange(0, T-dt, dt)

    # new mesh
    x_M = 6
    y_M = 0.2
    z_M = 0.53

    # 0.05 um above axon A (ECS)
    x_e = x_M; y_e = y_M + 0.1; z_e = z_M
    # mid point inside axon A (ICS)
    x_i = x_M; y_i = y_M - 0.1; z_i = z_M

    # trace concentrations
    phi_M, phi_M_ODE, E_Na, E_K = get_time_series_membrane(dt, T, fname, x_M, y_M, z_M)

    # bulk concentrations
    Na_e, K_e, Cl_e, _ = get_time_series(dt, T, fname, x_e, y_e, z_e)
    Na_i, K_i, Cl_i, _ = get_time_series(dt, T, fname, x_i, y_i, z_i)

    #################################################################
    # get data axons BC are stimulated

    # Concentration plots
    fig = plt.figure(figsize=(12*0.9,12*0.9))
    ax = plt.gca()

    ax1 = fig.add_subplot(3,3,1)
    plt.title(r'ECS Na$^+$')
    plt.plot(Na_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,2)
    plt.title(r'ECS K$^+$ )')
    plt.plot(K_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,3)
    plt.title(r'ECS Cl$^-$')
    plt.plot(Cl_e, linewidth=3, color='b')

    ax2 = fig.add_subplot(3,3,4)
    plt.title(r'Neuron Na$^+$')
    plt.plot(Na_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,5)
    plt.title(r'Neuron K$^+$')
    plt.plot(K_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,6)
    plt.title(r'Neuron Cl$^-$')
    plt.plot(Cl_i,linewidth=3, color='r')

    ax5 = fig.add_subplot(3,3,7)
    plt.title(r'Membrane potential neuron PDE')
    plt.plot(phi_M, linewidth=3)

    ax6 = fig.add_subplot(3,3,8)
    plt.title(r'Reversal potentials')
    plt.plot(E_K, linewidth=3)
    plt.plot(E_Na, linewidth=3)

    ax7 = fig.add_subplot(3,3,9)
    plt.title(r'Membrane potential neuron ODE')
    plt.plot(phi_M_ODE, linewidth=3)

    print("membrane potential", phi_M[0], phi_M[-1])
    print("membrane potential", phi_M_ODE[0], phi_M_ODE[-1])
    print("Na_e", Na_e[0], Na_e[-1])
    print("Na_i", Na_i[0], Na_i[-1])
    print("K_i", K_i[0], K_i[-1])
    print("K_e", K_e[0], K_e[-1])
    print("Cl_i", Cl_i[0], Cl_i[-1])
    print("Cl_e", Cl_e[0], Cl_e[-1])

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # save figure to file
    plt.savefig('results/figures/PDE_neuron.svg', format='svg')
    plt.close()

    return

def plot_3D_concentration_glial(res, T, dt, fname):

    time = np.arange(0, T-dt, dt)

    # new mesh
    x_M = 6
    y_M = 0.8
    z_M = 1.23

    # 0.05 um above axon A (ECS)
    x_e = x_M; y_e = y_M + 0.1; z_e = z_M
    # mid point inside axon A (ICS)
    x_i = x_M; y_i = y_M - 0.1; z_i = z_M

    # trace concentrations
    phi_M, phi_M_ODE, E_Na, E_K = get_time_series_membrane(dt, T, fname, x_M, y_M, z_M)

    # bulk concentrations
    Na_e, K_e, Cl_e, _ = get_time_series(dt, T, fname, x_e, y_e, z_e)
    Na_i, K_i, Cl_i, _ = get_time_series(dt, T, fname, x_i, y_i, z_i)

    # plot
    fig = plt.figure(figsize=(12*0.9,12*0.9))
    ax = plt.gca()

    ax1 = fig.add_subplot(3,3,1)
    plt.title(r'ECS Na$^+$')
    plt.plot(Na_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,2)
    plt.title(r'ECS K$^+$ )')
    plt.plot(K_e, linewidth=3, color='b')

    ax3 = fig.add_subplot(3,3,3)
    plt.title(r'ECS Cl$^-$')
    plt.plot(Cl_e, linewidth=3, color='b')

    ax2 = fig.add_subplot(3,3,4)
    plt.title(r'Glial Na$^+$')
    plt.plot(Na_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,5)
    plt.title(r'Glial K$^+$')
    plt.plot(K_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,6)
    plt.title(r'Glial Cl$^-$')
    plt.plot(Cl_i,linewidth=3, color='r')

    ax5 = fig.add_subplot(3,3,7)
    plt.title(r'Membrane potential glial PDE')
    plt.plot(phi_M, linewidth=3)

    ax6 = fig.add_subplot(3,3,8)
    plt.title(r'Reversal potentials')
    plt.plot(E_K, linewidth=3)
    plt.plot(E_Na, linewidth=3)

    ax7 = fig.add_subplot(3,3,9)
    plt.title(r'Membrane potential glial ODE')
    plt.plot(phi_M_ODE, linewidth=3)

    print("membrane potential", phi_M[0], phi_M[-1])
    print("membrane potential", phi_M_ODE[0] ,phi_M_ODE[-1])
    print("Na_e", Na_e[0], Na_e[-1])
    print("Na_i", Na_i[0], Na_i[-1])
    print("K_i", K_i[0], K_i[-1])
    print("K_e", K_e[0], K_e[-1])
    print("Cl_i", Cl_i[0], Cl_i[-1])
    print("Cl_e", Cl_e[0], Cl_e[-1])

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # save figure to file
    plt.savefig('results/figures/PDE_glial.svg', format='svg')
    plt.close()

    return

# create directory for figures
if not os.path.isdir('results/figures'):
    os.mkdir('results/figures')

# create figures
res_3D = '0' # mesh resolution for 3D axon bundle
dt = 0.1
T = 2

fname = 'results/data/calibration_two_tags/results.h5'

plot_3D_concentration_neuron(res_3D, T, dt, fname)
plot_3D_concentration_glial(res_3D, T, dt, fname)
