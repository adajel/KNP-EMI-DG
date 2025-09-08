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

def write_to_pvd(fname, fnameout, T, dt):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
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

    f_phi = XDMFFile(fnameout + 'pvd/pot.xdmf')
    f_Na = XDMFFile(fnameout + 'pvd/Na.xdmf')
    f_K = XDMFFile(fnameout + 'pvd/K.xdmf')
    f_Cl = XDMFFile(fnameout + 'pvd/Cl.xdmf')

    for n in range(1, int(T/dt)):

        print("n", n)

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


def get_time_series(dt, T, fname, x):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    x_e = x['E'][0]
    y_e = x['E'][1]
    z_e = x['E'][2]

    x_i = x['I'][0]
    y_i = x['I'][1]
    z_i = x['I'][2]

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    P1 = FiniteElement('DG', mesh.ufl_cell(), 1)
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

    """
    # define one facet to 10 for getting membrane potential
    for cell in cells(mesh):
        x_ = [cell.midpoint().x(), cell.midpoint().y(), cell.midpoint().z()]
        point_1 = 2700e-7 <= x_[0] <= 2900e-7 \
              and 1800e-7 <= x_[1] <= 2000e-7 \
              and 1900e-7 <= x_[2] <= 2100e-7

        if subdomains[cell] == 0 and point_1:
            print("ECS", x_[0], x_[1], x_[2])

        if subdomains[cell] == 2 and point_1:
            print("ICS", x_[0], x_[1], x_[2])

    sys.exit(0)
    """

    for n in range(1, int(T/dt)):
            # read file
            hdf5file.read(u, "/concentrations/vector_" + str(n))

            # K concentrations
            assign(f_K, u.sub(0))
            K_e.append(f_K(x_e, y_e, z_e))
            K_i.append(f_K(x_i, y_i, z_i))

            # Cl concentrations
            assign(f_Cl, u.sub(1))
            Cl_e.append(f_Cl(x_e, y_e, z_e))
            Cl_i.append(f_Cl(x_i, y_i, z_i))

            # Na concentrations
            hdf5file.read(v, "/elim_concentration/vector_" + str(n))
            assign(f_Na, v)
            Na_e.append(f_Na(x_e, y_e, z_e))
            Na_i.append(f_Na(x_i, y_i, z_i))

            # potential
            hdf5file.read(w, "/potential/vector_" + str(n))
            assign(f_phi, w)
            phi_e.append(f_phi(x_e, y_e, z_e))
            phi_i.append(f_phi(x_i, y_i, z_i))

    return Na_e, K_e, Cl_e, phi_e, Na_i, K_i, Cl_i, phi_i

def get_time_series_membrane_neuron(dt, T, fname, x, y, z):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    x_min = x - 1.0e-4; x_max = x + 1.0e-4
    y_min = y - 1.0e-3; y_max = y + 1.0e-3
    z_min = z; z_max = z + 1.0e-5

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        point_1 = y_min <= x[1] <= y_max \
              and x_min <= x[0] <= x_max \
              and z_min <= x[2] <= z_max

        """
        # ---------------------------------------------------
        # new ROI
        xmin = 2700e-7; xmax = 2900e-7
        ymin = 1800e-7; ymax = 2000e-7
        zmin = 1900e-7; zmax = 2100e-7

        if (xmin <= x[0] <= xmax) and \
           (ymin <= x[1] <= ymax) and \
           (zmin <= x[2] <= zmax) and (surfaces[facet] == 1):
            print("mark 10:", x[0], x[1], x[2])

        # ---------------------------------------------------

        """
        if point_1 and (surfaces[facet] == 1):
            print("mark 10:", x[0]*1.0e7, x[1]*1.0e7, x[2]*1.0e7)
            surfaces[facet] = 10
            break

    #sys.exit(0)

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


def get_time_series_membrane_glial(dt, T, fname, x, y, z):
    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    #x_min = x - 0.7e-5; x_max = x + 0.7e-5
    #y_min = y - 0.7e-4; y_max = y + 0.7e-4
    #z_min = z; z_max = z + 0.08e-5

    x_min = x - 1.0e-7; x_max = x + 1.0e-7
    y_min = y - 1.0e-6; y_max = y + 1.0e-6
    z_min = z; z_max = z + 1.0e-8

    # define one facet to 10 for getting membrane potential
    for facet in facets(mesh):
        x = [facet.midpoint().x(), facet.midpoint().y(), facet.midpoint().z()]
        point_1 = y_min <= x[1] <= y_max \
              and x_min <= x[0] <= x_max \
              and z_min <= x[2] <= z_max

        """
        # new ROI
        xmin = 2700e-7; xmax = 2900e-7
        ymin = 1800e-7; ymax = 2000e-7
        zmin = 1900e-7; zmax = 2100e-7

        if (xmin <= x[0] <= xmax) and \
           (ymin <= x[1] <= ymax) and \
           (zmin <= x[2] <= zmax) and (surfaces[facet] == 2):
            print("mark 10:", x[0], x[1], x[2])
        """

        if point_1 and (surfaces[facet] == 2):
            print("mark 10:", x[0]*1.0e7, x[1]*1.0e7, x[2]*1.0e7)
            surfaces[facet] = 10
            break

    #sys.exit(0)

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

def plot_3D_concentration_glial(res, T, dt, x, fname, fnameout):

    if not os.path.exists(fnameout):
        os.makedirs(fnameout)

    time = np.arange(0, T-dt, dt)

    x_M = x['M'][0]
    y_M = x['M'][1]
    z_M = x['M'][2]

    x_M_n = x['M_n'][0]
    y_M_n = x['M_n'][1]
    z_M_n = x['M_n'][2]

    # trace concentrations
    phi_M, phi_M_ODE, E_Na, E_K = get_time_series_membrane_glial(dt, T, fname, x_M, y_M, z_M)
    phi_M_n, _, _, _ = get_time_series_membrane_neuron(dt, T, fname, x_M_n, y_M_n, z_M_n)

    # bulk concentrations
    Na_e, K_e, Cl_e, _, Na_i, K_i, Cl_i, _  = get_time_series(dt, T, fname, x)

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

    ax7 = fig.add_subplot(3,3,9)
    plt.title(r'Membrane potential neuron')
    plt.plot(phi_M_n, linewidth=3)

    print("membrane potential", phi_M[0], phi_M[-1])
    print("membrane potential", phi_M_ODE[0], phi_M_ODE[-1])

    print("Na_e", Na_e[0], Na_e[-1])
    print("Na_i", Na_i[0], Na_i[-1])

    print("K_e", K_e[0], K_e[-1])
    print("K_i", K_i[0], K_i[-1])

    print("Cl_e", Cl_e[0], Cl_e[-1])
    print("Cl_i", Cl_i[0], Cl_i[-1])

    # make pretty
    ax.axis('off')
    plt.tight_layout()

    # save figure to file (NB! Delta [k] is saved)
    plt.savefig(fnameout + 'plot.svg', format='svg')

    f_phi_M = open(fnameout + 'phi_M.txt', "w")
    for p in phi_M:
        f_phi_M.write("%.10f \n" % p)
    f_phi_M.close()

    f_K_e = open(fnameout + 'K_ECS.txt', "w")
    for p in K_e:
        f_K_e.write("%.10f \n" % p)
    f_K_e.close()

    f_K_i = open(fnameout + 'K_ICS.txt', "w")
    for p in K_i:
        f_K_i.write("%.10f \n" % p)
    f_K_i.close()

    f_Na_e = open(fnameout + 'Na_ECS.txt', "w")
    for p in Na_e:
        f_Na_e.write("%.10f \n" % p)
    f_Na_e.close()

    f_Na_i = open(fnameout + 'Na_ICS.txt', "w")
    for p in Na_i:
        f_Na_i.write("%.10f \n" % p)
    f_Na_i.close()

    f_Cl_e = open(fnameout + 'Cl_ECS.txt', "w")
    for p in Cl_e:
        f_Cl_e.write("%.10f \n" % p)
    f_Cl_e.close()

    f_Cl_i = open(fnameout + 'Cl_ICS.txt', "w")
    for p in Cl_i:
        f_Cl_i.write("%.10f \n" % p)
    f_Cl_i.close()

    return


def plot_surface(fname, fnameout, T, dt):

    # read data file
    hdf5file = HDF5File(MPI.comm_world, fname, "r")

    mesh = Mesh()
    subdomains = MeshFunction("size_t", mesh, 2)
    surfaces = MeshFunction("size_t", mesh, 1)
    hdf5file.read(mesh, '/mesh', False)
    #mesh.coordinates()[:] *= 1e-7
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    ds = Measure('ds', domain=mesh, subdomain_data=surfaces)
    dS = Measure('dS', domain=mesh, subdomain_data=surfaces)

    surface_tags = (2,)
    #surface_tags = (1, 3)

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

    dlt = DltWriter(fnameout + 'pvd/bar', mesh, surface_mesh)
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
    #mesh.coordinates()[:] *= 1e-7
    hdf5file.read(subdomains, '/subdomains')
    hdf5file.read(surfaces, '/surfaces')

    ds = Measure('ds', domain=mesh, subdomain_data=surfaces)
    dS = Measure('dS', domain=mesh, subdomain_data=surfaces)

    surface_tags = (1, 2)

    surface_mesh, L, cell2Ldofs = P0surf_to_DLT0_map(surfaces, tags=surface_tags)

    P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)

    u = Function(V)
    uh = Function(V)

    # read file
    hdf5file.read(u, "/potential/vector_" + str(0))
    # K concentrations
    assign(uh, u)

    # To get the restriction as we want we build a P0(mesh) tagging function 
    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)

    V = FunctionSpace(mesh, 'CG', 0)
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

# create directory for figures
if not os.path.isdir('results/figures'):
    os.mkdir('results/figures')

# create figures
res_3D = '0' # mesh resolution for 3D axon bundle
dt = 0.1

x_ROI = {"M_n": [0.00028344048106604234, 0.00019829184694216556, 0.00020210869181408721],\
         "M": [0.0002807679444022024, 0.00018748192486428163, 0.0002003437621546238], \
         "E": [0.0002814348233730957, 0.0001951174445744194, 0.00020219069204765956], \
         "I": [0.00028564870433857667, 0.0001831684282583537, 0.0001980283625234914]}

# outside of input zone
x_outside_ROI = {"M": [0.0003844771569042325, 0.00038766154959819847, 0.00037534978610169353], \
                 "E": [0.0003637753007305006, 0.0003705360300998128, 0.0003625574818125377], \
                 "I": [0.0003548050620792965, 0.0003520433961759351, 0.0003739900092393671]}

T = 10

fname = "results/data/EMIx-synapse_100_tort_short_4_tort_both_ICS_and_ECS/results.h5"
fnameout = "results/data/EMIx-synapse_100_tort_short_4_tort_both_ICS_and_ECS/"
plot_3D_concentration_glial(res_3D, T, dt, x_ROI, fname, fnameout)
plot_surface(fname, fnameout, T, dt)
write_to_pvd(fname, fnameout, T, dt)

#fname = "results/data/EMIx-synapse_100_tort_short_2_tort_both_ICS_and_ECS/results.h5"
#fnameout = "results/data/EMIx-synapse_100_tort_short_2_tort_both_ICS_and_ECS/"
#plot_3D_concentration_glial(res_3D, T, dt, x_ROI, fname, fnameout)
#plot_surface(fname, fnameout, T, dt)
#write_to_pvd(fname, fnameout, T, dt)

#fname = "results/data/EMIx-synapse_100_tort_short_baseline/results.h5"
#fnameout = "results/data/EMIx-synapse_100_tort_short_baseline/"
#plot_3D_concentration_glial(res_3D, T, dt, x_ROI, fname, fnameout)
#plot_surface(fname, fnameout, T, dt)
#write_to_pvd(fname, fnameout, T, dt)

"""
T = 0.5
fname = "results/data/EMIx-synapse_100_tort_short_baseline_test/results.h5"
fnameout = "results/data/EMIx-synapse_100_tort_short_baseline_test/"
plot_3D_concentration_glial(res_3D, T, dt, x_ROI, fname, fnameout)
"""

#T = 60
#fname = "results/data/EMIx-synapse_100_tort_long_10_tort/results.h5"
#fnameout = "results/data/EMIx-synapse_100_tort_long_10_tort/"

#fname = "results/data/EMIx-synapse_100_tort_short_10_tort/results.h5"
#fnameout = "results/data/EMIx-synapse_100_tort_short_10_tort/"

#fname = "results/data/EMIx-synapse_100_tort_short_5_tort/results.h5"
#fnameout = "results/data/EMIx-synapse_100_tort_short_5_tort/"

#fname = "results/data/EMIx-synapse_100_tort_short_5_tort_both_ICS_and_ECS/results.h5"
#fnameout = "results/data/EMIx-synapse_100_tort_short_5_tort_both_ICS_and_ECS/"

"""
fname = "results/data/EMIx-synapse_100_tort_short_5_tort_both_ICS_and_ECS/results.h5"
fnameout = "results/data/EMIx-synapse_100_tort_short_5_tort_both_ICS_and_ECS/"

plot_surface(fname, fnameout, T, dt)
plot_3D_concentration_glial(res_3D, T, dt, x_ROI, fname, fnameout)
"""

#fnameout = 'results/data/EMIx-synapse_100_small_outside_ROI/'
#plot_3D_concentration_glial(res_3D, T, dt, x_outside_ROI, fname, fnameout)

### Compare 5 and 100 cells in ROI ###

#T = 2
#fname = 'results/data/EMIx-synapse_5/results.h5'
#fnameout = 'results/data/EMIx-synapse_5/glial_5'
#plot_3D_concentration_glial(res_3D, T, dt, x_ROI, fname, fnameout)

#T = 2
#fname = 'results/data/EMIx-synapse_100/results.h5'
#fnameout = 'results/data/EMIx-synapse_100/glial_100'
#plot_3D_concentration_glial(res_3D, T, dt, x_ROI, fname, fnameout)

### Compare 5 and 100 cells outside ROI ###

#T = 2
#fname = 'results/data/EMIx-synapse_5/results.h5'
#fnameout = 'results/data/EMIx-synapse_5/glial_5_outside_ROI'
#plot_3D_concentration_glial(res_3D, T, dt, x_outside_ROI, fname, fnameout)

#T = 2
#fname = 'results/data/EMIx-synapse_100/results.h5'
#fnameout = 'results/data/EMIx-synapse_100/glial_100_outside_ROI'
#plot_3D_concentration_glial(res_3D, T, dt, x_outside_ROI, fname, fnameout)

### Compare short and long stimuli in ROI  ###

#T = 25
#fname = 'results/data/EMIx-synapse_100_tort_long/results.h5'
#fnameout = 'results/data/EMIx-synapse_100_tort_long/'
#plot_3D_concentration_glial(res_3D, T, dt, x_ROI, fname, fnameout)

#fname = 'results/data/EMIx-synapse_100_tort_short/results.h5'
#fnameout = 'results/data/EMIx-synapse_100_tort_short/'
#plot_3D_concentration_glial(res_3D, T, dt, x_ROI, fname, fnameout)

### Compare short and long stimuli outside of ROI  ###
#T = 25
#fname = 'results/data/EMIx-synapse_100_tort_long/results.h5'
#fnameout = 'results/data/EMIx-synapse_100_tort_long_outside/'
#plot_3D_concentration_glial(res_3D, T, dt, x_outside_ROI, fname, fnameout)

#fname = 'results/data/EMIx-synapse_100_tort_short/results.h5'
#fnameout = 'results/data/EMIx-synapse_100_tort_short_outside/'
#plot_3D_concentration_glial(res_3D, T, dt, x_outside_ROI, fname, fnameout)

### Compare dt ###
#T = 2.0
#dt = 0.2
#fname = 'results/data/EMIx-synapse_100_dt_coarse/results.h5'
#fnameout = 'results/data/EMIx-synapse_100_dt_coarse/'
#plot_3D_concentration_glial(res_3D, T, dt, x_outside_ROI, fname, fnameout)

#T = 2.0
#dt = 0.1
#fname = 'results/data/EMIx-synapse_100_dt_fine/results.h5'
#fnameout = 'results/data/EMIx-synapse_100_dt_fine/'
#plot_3D_concentration_glial(res_3D, T, dt, x_outside_ROI, fname, fnameout)


#plot_surface(fname, fnameout, T, dt)


#plot_3D_concentration_neuron(res_3D, T, dt, fname)

#get_velocity(fname, T, dt)
#plot_surface_time(fname, T, dt)
