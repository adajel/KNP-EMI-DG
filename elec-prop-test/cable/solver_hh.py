from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

nx = 1500
len = 305e-4 # 305 um
mesh = IntervalMesh(nx, 0, len)

V = FunctionSpace(mesh, "CG", 1)
u_PDE = TrialFunction(V)
v = TestFunction(V)

w = Function(V)

# initial states
u_prev = Function(V)
m_prev = Function(V)
n_prev = Function(V)
h_prev = Function(V)

t = Constant(0)
dt = 0.1           # ms
Tstop = 30.0       # ms

# initial membrane potential (mV)
u_prev.assign(Constant(-65))
n_prev.assign(Constant(0.27622914792))
m_prev.assign(Constant(0.0379183462722))
h_prev.assign(Constant(0.688489218108))

w_ = 6.0e-5                 # cm
sigma_i = 20.12             # ICS conductance mS/cm
delta = sigma_i * w_ / 4

# Membrane properties
g_K_bar = 36.0
g_Na_bar = 120.0
g_leak = 0.4
C_M = 1.0
E_K = -88.98
E_Na = 54.81
E_leak = -54

def alpha_m(V):
    """Channel gating kinetics. Functions of membrane voltage"""
    return 0.1*(V+40.0)/(1.0 - exp(-(V+40.0) / 10.0))

def beta_m(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*exp(-(V+65.0) / 18.0)

def alpha_h(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*exp(-(V+65.0) / 20.0)

def beta_h(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + exp(-(V+35.0) / 10.0))

def alpha_n(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - exp(-(V+55.0) / 10.0))

def beta_n(V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*exp(-(V+65) / 80.0)

a = C_M * inner(u_PDE, v) * dx + dt * delta * inner(grad(u_PDE), grad(v)) * dx
L = C_M * u_prev * v * dx \

phi_M = []; time = []
n_M = []; m_M = []; h_M = []

# set stimulus PDE
g_syn_bar = 10
g_syn = Expression('g_syn_bar * exp(-fmod(t, 20.0)/2.0) * (x[0] < 20.0e-4)', \
                   t=t, g_syn_bar=g_syn_bar, degree=4)

# total conductances
g_Na = g_Na_bar * m_prev ** 3 * h_prev  # sodium
g_K = g_K_bar * n_prev ** 4             # potassium

# total ionic current
I_ion = g_K * (u_prev - E_K) \
      + g_Na * (u_prev - E_Na) \
      + g_leak * (u_prev - E_leak) \
      + g_syn * (u_prev - E_Na)

# rhs ODEs in HH system
n_rhs = alpha_n(u_prev) * (1.0 - n_prev) - beta_n(u_prev) * n_prev
m_rhs = alpha_m(u_prev) * (1.0 - m_prev) - beta_m(u_prev) * m_prev
h_rhs = alpha_h(u_prev) * (1.0 - h_prev) - beta_h(u_prev) * h_prev
u_rhs = - I_ion

# ODE time step
dt_ODE = dt/20.

p1 = 20
p2 = 40
t1 = 0
t2 = 0

u_0 = Function(V)
u_1 = Function(V)
u_2 = Function(V)
u_3 = Function(V)
u_4 = Function(V)
u_5 = Function(V)
u_6 = Function(V)
u_7 = Function(V)


for k in range(int(round(Tstop/dt))):

    # solve for ODEs
    for i in range(20):
        n = n_rhs * dt_ODE + n_prev
        m = m_rhs * dt_ODE + m_prev
        h = h_rhs * dt_ODE + h_prev
        u_ODE = u_rhs / C_M * dt_ODE + u_prev

        # update u and gating
        u_prev.assign(project(u_ODE, V))
        n_prev.assign(project(n, V))
        m_prev.assign(project(m, V))
        h_prev.assign(project(h, V))

    # solve PDE system
    solve(a == L, w)
    u_prev.assign(w) # update u

    if float(t) == 0:
        u_0.assign(w)
        print("0")
    elif 1.0 <= float(t) <= 1.1:
        u_1.assign(w)
        print("1")
    elif 5.0 <= float(t) <= 5.1:
        u_2.assign(w)
        print("2")
    elif 9.0 <= float(t) <= 9.1:
        u_3.assign(w)
        print("3")
    elif 12.0 <= float(t) <= 12.1:
        u_4.assign(w)
        print("4")

    # save for plot
    point = 100e-4
    phi_M.append(u_prev(point))
    time.append(float(t))
    m_M.append(m_prev(point))
    n_M.append(n_prev(point))
    h_M.append(h_prev(point))

    if (u_prev(p1*1.0e-4) > 20 and t1 == 0):
        t1 = float(t)
    if (u_prev(p2*1.0e-4) > 20 and t2 == 0):
        t2 = float(t)

    # update time
    t.assign(t + dt)
    g_syn.t = t

header = "x y1"
phi_dat = [phi_M]
phi_dat.insert(0, time)
a_phi = np.asarray(phi_dat)
np.savetxt("../results/figures/phi_time_cable.dat", np.transpose(a_phi), delimiter=" ", header=header)

#delta_t = t2 - t1
#delta_x = p2 - p1

#print("velocity (um/ms) = ", delta_x/delta_t)

plt.figure()
plot(u_0, label="init")
plot(u_1, label="1 s")
plot(u_2, label="5 s")
plot(u_3, label="9 s")
plot(u_4, label="12 s")
plot(w, label="end time")
plt.legend()
plt.ylim(-100, 60)
plt.savefig("phi_space.png")
plt.close()

def plot_membrane_space(fname, n):

    plt.figure(figsize=(4.6, 3.3))
    plt.xlabel(r"x-position ($\mu$m)")
    plt.ylabel("$\phi_M$ (mV)")
    plt.ylim([-80, -60])
    #plt.ylim([-100e-3, -70e-3])
    #plt.yticks([-80, -60, -40, -20, 0, 20, 40])
    #plt.ylim([-90, 50])

    colors_n = [c_1, c_2, c_3]

    header = "x r0 r1 r2"
    phi_dat = []

    for idx, n in enumerate([1, 50, 99]):
        phi_M_n = get_plottable_gamma_function(fname, n)
        n_new = n/10. # convert ns to ms
        x = np.linspace(5, 305, len(phi_M_n))
        plt.plot(x, phi_M_n, linewidth=2.5, color=colors_n[idx], label=r"%d ms" % n_new)
        phi_dat.append(phi_M_n)

    plt.legend()
    #plt.tight_layout()
    # save figure
    plt.savefig("results/figures/phi_M_space.png")
    plt.close()

    phi_dat.insert(0, x)
    a_phi = np.asarray(phi_dat)
    np.savetxt("results/figures/phi_space_knp_emi.dat", np.transpose(a_phi), delimiter=" ", header=header)

    return

plt.figure()
plt.plot(phi_M)
plt.ylim(-100, 50)
plt.savefig("phi_time.png")
plt.show()
plt.close()

plt.figure()
plot(m, label="m")
plot(n, label="n")
plot(h, label="h")
plt.ylim(0, 1)
plt.legend()
plt.savefig("gating_hh.png")
plt.close()

plt.figure()
plt.plot(m_M, label="m")
plt.plot(n_M, label="n")
plt.plot(h_M, label="h")
plt.ylim(0, 1)
plt.legend()
plt.savefig("gating_time_hh.png")
plt.close()
