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

from surf_plot.vtk_io import DltWriter
from surf_plot.dlt_embedding import P0surf_to_DLT0_map

JUMP = lambda f, n: minus(f, n) - plus(f, n)

# set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

path = 'results/data/'

def write_to_pvd(dt, T, fname, ns):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    #mesh.coordinates()[:] *= 1e6
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    P1 = FiniteElement('DG', mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, MixedElement(2*[P1]))
    V = FunctionSpace(mesh, P1)

    u = Function(W)
    v = Function(V)
    w = Function(V)

    Na = Function(V)
    K = Function(V)
    Cl = Function(V)
    phi = Function(V)

    filename = 'results/data/rat_neuron/'

    f_phi = XDMFFile(filename + 'pvd/pot.xdmf')
    f_Na = XDMFFile(filename + 'pvd/Na.xdmf')
    f_K = XDMFFile(filename + 'pvd/K.xdmf')
    f_Cl = XDMFFile(filename + 'pvd/Cl.xdmf')

    for n in ns:

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

        f_phi.write_checkpoint(phi, "phi", time_step=n, append=True)
        f_Na.write_checkpoint(Na, "Na_", time_step=n, append=True)
        f_K.write_checkpoint(K, "K_", time_step=n, append=True)
        f_Cl.write_checkpoint(Cl, "Cl_", time_step=n, append=True)

    f_phi.close()
    f_Na.close()
    f_K.close()
    f_Cl.close()

    return


def get_time_series(dt, T, fname, x, y, z):
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
            phi.append(1.0e3*f_phi(x, y, z))

    return Na, K, Cl, phi

def get_time_series_membrane(dt, T, fname, x, y, z, tag):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    x_min = x - 0.1; x_max = x + 0.1
    y_min = y - 0.1; y_max = y + 0.1
    z_min = z - 0.1; z_max = z + 0.1

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        point_1 = y_min <= x[1] <= y_max \
              and x_min <= x[0] <= x_max \
              and z_min <= x[2] <= z_max
        if point_1 and surfaces[facet] == tag:
            print(x[0], x[1], x[2])
            surfaces[facet] = 10
            break

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

def get_time_series_gating(dt, T, fname, x, y, z):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    x_min = x - 0.1; x_max = x + 0.1
    y_min = y - 0.1; y_max = y + 0.1
    z_min = z - 0.1; z_max = z + 0.1

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        point_1 = y_min <= x[1] <= y_max \
              and x_min <= x[0] <= x_max \
              and z_min <= x[2] <= z_max
        if point_1:
            print(x[0], x[1], x[2])
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


def plot_3D_concentration(res, T, dt):

    temperature = 300 # temperature (K)
    F = 96485         # Faraday's constant (C/mol)
    R = 8.314         # Gas constant (J/(K*mol))

    time = 1.0e3*np.arange(0, T-dt, dt)

    # dendrite membrane point
    x_M_D = -13.409898613569093; y_M_D = -75.18635834947439; z_M_D =  20.947389697500906

    # soma / axon membrane point
    #x_M_A = -2.2763068484050986; y_M_A = 30.793095066405495; z_M_A = -2.063150759041879
    x_M_A = 10.847329267445332; y_M_A = -3.630288586838453; z_M_A = -2.063150759041879
    # 0.05 um above axon A (ECS)
    x_e_A = x_M_A + 0.2; y_e_A = y_M_A; z_e_A = z_M_A
    # mid point inside axon A (ICS)
    x_i_A = x_M_A - 0.2; y_i_A = y_M_A; z_i_A = z_M_A

    #################################################################
    # get data axon A is stimulated
    fname = 'results/data/rat_neuron/results.h5'

    # trace concentrations at soma
    phi_M_a, E_Na_a, E_K_a = get_time_series_membrane(dt, T, fname, x_M_A, y_M_A, z_M_A, 2)

    # bulk concentrations
    Na_e, K_e, Cl_e, _ = get_time_series(dt, T, fname, x_e_A, y_e_A, z_e_A)
    Na_i, K_i, Cl_i, _ = get_time_series(dt, T, fname, x_i_A, y_i_A, z_i_A)

    # trace concentrations at dendrite
    phi_M_d, E_Na_d, E_K_d = get_time_series_membrane(dt, T, fname, x_M_D, y_M_D, z_M_D, 1)

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
    plt.title(r'Membrane potential axon')
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(phi_M_a, linewidth=3)

    ax5 = fig.add_subplot(3,3,8)
    plt.title(r'Membrane potential dendrite')
    plt.ylabel(r'$\phi_M$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(phi_M_d, linewidth=3)

    ax6 = fig.add_subplot(3,3,9)
    plt.title(r'Na$^+$ reversal potential')
    plt.ylabel(r'E$_Na$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(E_K_a, linewidth=3)
    plt.plot(E_Na_a, linewidth=3)
    plt.plot(E_K_d, linewidth=3)
    plt.plot(E_Na_d, linewidth=3)

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # save figure to file
    plt.savefig('results/figures/pot_con_rat_3D.svg', format='svg')

    f_phi_M = open('phi_M_3D.txt', "w")
    for p in phi_M_a:
        f_phi_M.write("%.10f \n" % p)
    f_phi_M.close()

    return

def plot_surface(fname, T, dt):

    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    ds = Measure('ds', domain=mesh, subdomain_data=surfaces)
    dS = Measure('dS', domain=mesh, subdomain_data=surfaces)

    surface_tags = (1, 2)

    surface_mesh, L, cell2Ldofs = P0surf_to_DLT0_map(surfaces, tags=surface_tags)

    P1 = FiniteElement('DG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)

    u = Function(V)
    uh = Function(V)

    # read file
    n = 0
    hdf5file.read(u, "/potential/vector_" + str(n))

    # K concentrations
    assign(uh, u)

    # To get the restriction as we want we build a P0(mesh) tagging function 
    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)

    V = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(V)

    subdomain_tag = Function(V)
    hK = CellVolume(mesh)
    # Project coefs to P0 based on subdomain
    assemble((1/hK)*inner(v, Constant(1))*dx(1) + (1/hK)*inner(v, Constant(2))*dx(2), subdomain_tag.vector())
    # With the cell taggging function we will pick sides based on the marker value
    big_side = lambda u, K=subdomain_tag: conditional(gt(K('+'), K('-')), u('+'), u('-'))
    small_side = lambda u, K=subdomain_tag: conditional(le(K('+'), K('-')), u('+'), u('-'))

    # So finally for the projection of P0 data
    dl = TestFunction(L)
    fK = avg(FacetArea(mesh))
    # To get the quantity on the surface
    vh = Function(L)

    fK = FacetArea(mesh)
    dS0 = dS.reconstruct(metadata={'quadrature_degree': 0})
    ds0 = ds.reconstruct(metadata={'quadrature_degree': 0})

    #for truth, side in enumerate((big_side, small_side), 1):
    #print(side)
    # NOTE: that DLT is "continuous" on the facet, avg is just to shout up FFC
    #assemble(sum((1.0e3*1/avg(fK))*inner(small_side(uh) - big_side(uh), avg(dl))*dS0(tag) 
                    #for tag in surface_tags), tensor=vh.vector())
    #as_backend_type(vh.vector()).update_ghost_values()
    L_form = sum((1.0e3*1/avg(fK))*inner(small_side(uh) - big_side(uh), avg(dl))*dS0(tag) 
                    for tag in surface_tags)

    assemble(L_form, tensor=vh.vector())
    as_backend_type(vh.vector()).update_ghost_values()

    values = vh.vector().get_local()[cell2Ldofs]
    cell_data = {'uh': values}

    dlt = DltWriter('testing/bar', mesh, surface_mesh)
    with dlt as output:
        output.write(cell_data, t=0)

        # Some more time steps
        for n in range(1, int(T/dt)):
            # read file
            hdf5file.read(u, "/potential/vector_" + str(n))
            # K concentrations
            assign(uh, u)

            assemble(L_form, tensor=vh.vector())
            as_backend_type(vh.vector()).update_ghost_values()

            # Now we build the data for P0 function on the mesh
            values[:] = vh.vector().get_local()[cell2Ldofs]
            print(np.linalg.norm(values))
            output.write(cell_data, t=n)

    # Check the data
    #true = truth*np.ones_like(values)
    #assert np.linalg.norm(true - values) < 1E-12

    return

def plot_surface_time(fname, T, dt):

    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    ds = Measure('ds', domain=mesh, subdomain_data=surfaces)
    dS = Measure('dS', domain=mesh, subdomain_data=surfaces)

    surface_tags = (1, 2)

    surface_mesh, L, cell2Ldofs = P0surf_to_DLT0_map(surfaces, tags=surface_tags)

    P1 = FiniteElement('DG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)

    u = Function(V)
    uh = Function(V)

    # read file
    hdf5file.read(u, "/potential/vector_" + str(0))
    # K concentrations
    assign(uh, u)

    # To get the restriction as we want we build a P0(mesh) tagging function 
    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)

    V = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(V)

    subdomain_tag = Function(V)
    hK = CellVolume(mesh)
    # Project coefs to P0 based on subdomain
    assemble((1/hK)*inner(v, Constant(1))*dx(1) + (1/hK)*inner(v, Constant(2))*dx(2), subdomain_tag.vector())
    # With the cell taggging function we will pick sides based on the marker value
    big_side = lambda u, K=subdomain_tag: conditional(gt(K('+'), K('-')), u('+'), u('-'))
    small_side = lambda u, K=subdomain_tag: conditional(le(K('+'), K('-')), u('+'), u('-'))

    # So finally for the projection of P0 data
    dl = TestFunction(L)
    fK = avg(FacetArea(mesh))
    # To get the quantity on the surface
    vh = Function(L)

    fK = FacetArea(mesh)
    dS0 = dS.reconstruct(metadata={'quadrature_degree': 0})
    ds0 = ds.reconstruct(metadata={'quadrature_degree': 0})

    #for truth, side in enumerate((big_side, small_side), 1):
    #print(side)
    # NOTE: that DLT is "continuous" on the facet, avg is just to shout up FFC
    L_form = sum((1.0e3*1/avg(fK))*inner(small_side(uh) - big_side(uh), avg(dl))*dS0(tag) 
                    for tag in surface_tags)

    assemble(L_form, tensor=vh.vector())
    as_backend_type(vh.vector()).update_ghost_values()

    # At this point vh has the quantitiy of interest. Iguess it typically is
    # some restriction, see `example_P0.py` for how to get them
    # For plotting we always want grab subset of DLT dofs
    values = vh.vector().get_local()[cell2Ldofs]

    # And dump - note that we can have several quantities (e.g. neg_ih) in the
    # same file
    cell_data = {'uh': values, 'neg_uh': values}

    dlt = DltWriter(f'testing/cux_world{mesh.mpi_comm().size}', mesh, surface_mesh)
    with dlt as output:
        output.write(cell_data, t=0)

        # Some more time steps
        for n in range(1, int(T/dt)):
            # read file
            hdf5file.read(u, "/potential/vector_" + str(n))
            # potential
            assign(uh, u)

            assemble(L_form, tensor=vh.vector())
            as_backend_type(vh.vector()).update_ghost_values()

            # Now we build the data for P0 function on the mesh
            values[:] = vh.vector().get_local()[cell2Ldofs]
            print(np.linalg.norm(values))
            output.write(cell_data, t=n)
    return

def get_velocity(fname, T, dt):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e6
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    #x_min = 0.1; x_max = x + 0.1
    #y_min = 0.1; y_max = y + 0.1
    #z_min = 0.1; z_max = z + 0.1

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        # mark top point
        if surfaces[facet] == 2 and (x[1] > 539):
            print(x[0], x[1], x[2])
            surfaces[facet] = 10
        # mark bottom point
        if surfaces[facet] == 2 and (-0.004 < x[1] < 0.0045):
            print(x[0], x[1], x[2])
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

            if 1.0e3*assemble(1.0/iface_size_10*avg(phi_M)*dS(10)) > 0 and time_point_1 == 0:
                time_point_1 = n

            if 1.0e3*assemble(1.0/iface_size_10*avg(phi_M)*dS(20)) > 0 and time_point_2 == 0:
                time_point_2 = n

            print(time_point_1)
            print(time_point_2)

            # if membrane potential has reach 0 in both points then break
            if (time_point_1 > 0) and (time_point_2 > 0):
                break

    delta_t = (time_point_2 - time_point_1)*1.0e-4 # s
    delta_x = 539e-6 # m

    print("velocity (m/s)", delta_x/delta_t)

    return phi_M_s_1, phi_M_s_2

# create directory for figures
if not os.path.isdir('results/figures'):
    os.mkdir('results/figures')

# create figures
res_3D = '0' # mesh resolution for 3D axon bundle
fname = 'results/data/rat_neuron/results.h5'

dt = 1.0e-4
T = 5.0e-2

#ns = range(1, int(T/dt))
#ns = (370, 380, 390, 400, 410, 420, 430, 440, 450)
#ns = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
#write_to_pvd(dt, T, fname, ns)

#print("plot 3D concentration")
#plot_3D_concentration(res_3D, T, dt)
get_velocity(fname, T, dt)

#plot_surface(fname, T, dt)
#plot_surface_time(fname, T, dt)
