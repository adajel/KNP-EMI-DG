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
        if point_1 and surfaces[facet] == 1:
        #if surfaces[facet] == 1:
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
    phi_M_ODE_s = []
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

            hdf5file.read(w_phi_ODE, "/membranepotential/vector_" + str(n))
            assign(f_phi_ODE, w_phi_ODE)
            phi_M_ODE_s.append(assemble(1.0/iface_size*avg(f_phi_ODE)*dS(10)))

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

    return phi_M_s, phi_M_ODE_s, E_Na_s, E_K_s

def plot_3D_concentration(res, T, dt, fname):

    time = np.arange(0, T-dt, dt)

    # original mesh
    #x_M = 20
    #y_M = 0.2
    #z_M = 0.63

    # new mesh
    x_M = 8
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

    #################################################################
    # get data axons BC are stimulated

    # Concentration plots
    fig = plt.figure(figsize=(12*0.9,12*0.9))
    ax = plt.gca()

    ax1 = fig.add_subplot(3,3,1)
    plt.title(r'Na$^+$ concentration (ECS)')
    plt.ylabel(r'[Na]$_e$ (mM)')
    #plt.ylim([90, 110])
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
    #plt.ylim([99, 101])
    plt.plot(Na_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,5)
    plt.title(r'K$^+$ concentration (ICS)')
    plt.ylabel(r'[K]$_i$ (mM)')
    #plt.ylim([4, 4.2])
    plt.plot(K_i,linewidth=3, color='r')

    ax2 = fig.add_subplot(3,3,6)
    plt.title(r'Cl$^-$ concentration (ICS)')
    plt.ylabel(r'[Cl]$_i$ (mM)')
    #plt.ylim([103, 105])
    plt.plot(Cl_i,linewidth=3, color='r')

    ax5 = fig.add_subplot(3,3,7)
    plt.title(r'Membrane potential')
    plt.ylabel(r'$\phi_M$ (mV)')
    #plt.ylim([-71.3, -71.2])
    plt.xlabel(r'time (ms)')
    plt.plot(phi_M, linewidth=3)

    ax6 = fig.add_subplot(3,3,8)
    plt.title(r'Na$^+$ reversal potential')
    plt.ylabel(r'E$_Na$ (mV)')
    plt.xlabel(r'time (ms)')
    plt.plot(E_K, linewidth=3)
    plt.plot(E_Na, linewidth=3)

    ax7 = fig.add_subplot(3,3,9)
    plt.title(r'Membrane potential ODE')
    #plt.ylim([-71.3, -71.2])
    plt.plot(phi_M_ODE, linewidth=3)

    print("membrane potential", phi_M[-1])
    print("membrane potential", phi_M_ODE[-1])
    #print("Na_e", Na_e[-1])
    #print("Na_i", Na_i[-1])
    #print("K_i", K_i[-1])
    #print("K_e", K_e[-1])
    #print("Cl_i", Cl_i[-1])
    #print("Cl_e", Cl_e[-1])

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # save figure to file
    plt.savefig('results/figures/callibrate.svg', format='svg')

    f_phi_M = open('phi_M_3D.txt', "w")
    for p in phi_M:
        f_phi_M.write("%.10f \n" % p*1000)
    f_phi_M.close()

    return

def plot_surface(fname, T, dt):

    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    mesh.coordinates()[:] *= 1e7
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
    small_side = lambda u, K=subdomain_tag: conditional(gt(K('+'), K('-')), u('+'), u('-'))
    big_side = lambda u, K=subdomain_tag: conditional(le(K('+'), K('-')), u('+'), u('-'))

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
    L_form = sum((1/avg(fK))*inner(small_side(uh) - big_side(uh), avg(dl))*dS0(tag) 
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
    mesh.coordinates()[:] *= 1e7
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
    small_side = lambda u, K=subdomain_tag: conditional(gt(K('+'), K('-')), u('+'), u('-'))
    big_side = lambda u, K=subdomain_tag: conditional(le(K('+'), K('-')), u('+'), u('-'))

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
    L_form = sum((1/avg(fK))*inner(small_side(uh) - big_side(uh), avg(dl))*dS0(tag) 
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
dt = 0.1
T = 10

#fname = 'results/data/EMIX/results.h5'
#plot_surface(fname, T, dt)
#plot_surface_time(fname, T, dt)

fname = 'results/data/calibration/results.h5'
plot_3D_concentration(res_3D, T, dt, fname)

#write_to_pvd(dt, T, fname)
#get_velocity(fname, T, dt)
