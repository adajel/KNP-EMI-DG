from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

a = 0
b = 50e-4 # cm
n = 100
mesh = IntervalMesh(n, a, b)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

w = Function(V)
u_prev = Function(V)

t = Constant(0)
dt = Constant(0.1)      # ms
Tstop = 10              # ms

C_M = Constant(1.0)     # capacitance ()

# stimuli current
I_stim = 400
I_e = Expression('I_stim * (x[0] < 5e-4)', degree=4, I_stim=I_stim)
# initial membrane potential (mV)
u_prev.assign(Constant(-68))

w_ = 6.0e-5         # cm
sigma_i = 20        # S/cm
E_l = -47.2         # mV
g_l = 50            # mS/cm²

delta = sigma_i * w_ / 4

m_K = 4.0           # threshold ECS K (mol/m^3)
m_Na = 12.0         # threshold ICS Na (mol/m^3)
I_max = 130         # max pump strength (A/m^2)
K_e = 4.0           # threshold ECS K (mol/m^3)
Na_i = 12.0         # threshold ICS Na (mol/m^3)

I_pump = I_max / ((1 + m_K / K_e) ** 2 * (1 + m_Na / Na_i) ** 3)

a = C_M / dt * inner(u, v) * dx \
  + delta * inner(grad(u), grad(v)) * dx \
  + g_l * u  * v * dx
L = C_M / dt * u_prev * v * dx \
  + (g_l * E_l + I_pump) * v * dx + I_e * v * dx

for k in range(int(round(Tstop/float(dt)))):
    # solve system
    #solve(a, w, L)
    solve(a == L, w)

    # update u
    u_prev.assign(w)
    # update time
    t.assign(float(t + dt))

plt.figure()
plot(w)
plt.ylim(-100, 20)
plt.savefig("cable.png")
