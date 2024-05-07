from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

nx = 1000
#len = 500e-4 # 200 um
len = 110e-4 # 200 um
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

u_0 = Function(V)
u_1 = Function(V)
u_2 = Function(V)
u_3 = Function(V)
u_4 = Function(V)
u_5 = Function(V)
u_6 = Function(V)
u_7 = Function(V)

t = Constant(0)
dt = 0.01           # ms
Tstop = 50.0        # ms

# initial membrane potential (mV)
#u_prev.assign(Constant(-68.318))
u_prev.assign(Constant(-65))
n_prev.assign(Constant(0.27622914792))
m_prev.assign(Constant(0.0379183462722))
h_prev.assign(Constant(0.688489218108))

w_ = 6.0e-5         # cm

sigma_i = 20.12     # mS/cm
delta = sigma_i * w_ / 4

# Average potassium channel conductance per unit area (mS/cm^2)
g_K_bar = 36.0
g_Na_bar = 120.0
#g_leak = 0.25
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

phi_M = []
n_M = []
m_M = []
h_M = []

# Set stimulus PDE
g_syn_bar = 10
g_syn = Expression('g_syn_bar * exp(-fmod(t, 20.0)/2.0) * (x[0] < 20.0e-4)', \
                   t=t, g_syn_bar=g_syn_bar, degree=4)

g_Na = g_Na_bar * m_prev ** 3 * h_prev
g_K = g_K_bar * n_prev ** 4

I_ion = g_K * (u_prev - E_K) \
      + g_Na * (u_prev - E_Na) \
      + g_leak * (u_prev - E_leak) \
      + g_syn * (u_prev - E_Na)

n_rhs = alpha_n(u_prev) * (1.0 - n_prev) - beta_n(u_prev) * n_prev
m_rhs = alpha_m(u_prev) * (1.0 - m_prev) - beta_m(u_prev) * m_prev
h_rhs = alpha_h(u_prev) * (1.0 - h_prev) - beta_h(u_prev) * h_prev
u_rhs = - I_ion

dt_ODE = dt/20.

p1 = 20
#p2 = 120
p2 = 40
t1 = 0
t2 = 0

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
    elif 0.5 <= float(t) <= 0.6:
        u_1.assign(w)
        print("1")
    elif 0.9 <= float(t) <= 1.0:
        u_2.assign(w)
        print("2")
    elif 1.2 <= float(t) <= 1.3:
        u_3.assign(w)
        print("3")
    elif 1.5 <= float(t) <= 1.6:
        u_4.assign(w)
        print("4")
    elif 1.8 <= float(t) <= 1.9:
        u_5.assign(w)
        print("4")
    elif 2.0 <= float(t) <= 2.1:
        u_6.assign(w)
        print("4")
    elif 2.4 <= float(t) <= 2.5:
        u_7.assign(w)
        print("4")

    # save for plot
    point = 100e-4
    phi_M.append(u_prev(point))
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

#delta_t = t2 - t1
#delta_x = p2 - p1

#print("velocity (um/ms) = ", delta_x/delta_t)

plt.figure()
plot(u_0, label="init")
plot(u_1, label="1 s")
plot(u_2, label="2 s")
plot(u_3, label="3 s")
plot(u_4, label="4 s")
plot(u_5, label="4 s")
plot(u_6, label="4 s")
plot(u_7, label="4 s")
plot(w, label="end time")
plt.legend()
plt.ylim(-100, 60)
plt.savefig("phi_space.png")
plt.close()

plt.figure()
plt.plot(phi_M)
plt.ylim(-100, 50)
plt.savefig("phi_time.png")
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
