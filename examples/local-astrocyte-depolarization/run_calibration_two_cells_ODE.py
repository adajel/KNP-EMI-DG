import os
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import mm_two_tags_calibration_ODE as ode
from knpemidg.membrane import MembraneModel
from collections import namedtuple

mesh = df.UnitSquareMesh(2, 2)
V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)

facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
tag = 0

g_syn_bar = 0
stimulus = {'stim_amplitude': g_syn_bar}

membrane = MembraneModel(ode, facet_f=facet_f, tag=tag, V=V)

V_index_n = ode.state_indices('V_n')
V_index_g = ode.state_indices('V_g')
K_e_index = ode.state_indices('K_e')
K_n_index = ode.state_indices('K_n')
K_g_index = ode.state_indices('K_g')
Na_e_index = ode.state_indices('Na_e')
Na_n_index = ode.state_indices('Na_n')
Na_g_index = ode.state_indices('Na_g')
Cl_e_index = ode.state_indices('Cl_e')
Cl_n_index = ode.state_indices('Cl_n')
Cl_g_index = ode.state_indices('Cl_g')

n_index = ode.state_indices('n')
m_index = ode.state_indices('m')
h_index = ode.state_indices('h')

potential_history_n = []
potential_history_g = []
K_e_history = []
K_n_history = []
K_g_history = []
Na_e_history = []
Na_n_history = []
Na_g_history = []
Cl_e_history = []
Cl_n_history = []
Cl_g_history = []

n_history = []
m_history = []
h_history = []

#for _ in range(50000):
for _ in range(200000):
    membrane.step_lsoda(dt=0.1, stimulus=stimulus)

    potential_history_n.append(1*membrane.states[:, V_index_n])
    potential_history_g.append(1*membrane.states[:, V_index_g])
    K_e_history.append(1*membrane.states[:, K_e_index])
    K_n_history.append(1*membrane.states[:, K_n_index])
    K_g_history.append(1*membrane.states[:, K_g_index])
    Na_e_history.append(1*membrane.states[:, Na_e_index])
    Na_n_history.append(1*membrane.states[:, Na_n_index])
    Na_g_history.append(1*membrane.states[:, Na_g_index])
    Cl_e_history.append(1*membrane.states[:, Cl_e_index])
    Cl_n_history.append(1*membrane.states[:, Cl_n_index])
    Cl_g_history.append(1*membrane.states[:, Cl_g_index])

    n_history.append(1*membrane.states[:, n_index])
    m_history.append(1*membrane.states[:, m_index])
    h_history.append(1*membrane.states[:, h_index])

potential_history_n = np.array(potential_history_n)
potential_history_g = np.array(potential_history_g)
K_e_history = np.array(K_e_history)
K_n_history = np.array(K_n_history)
K_g_history = np.array(K_g_history)
Na_e_history = np.array(Na_e_history)
Na_n_history = np.array(Na_n_history)
Na_g_history = np.array(Na_g_history)
Cl_e_history = np.array(Cl_e_history)
Cl_n_history = np.array(Cl_n_history)
Cl_g_history = np.array(Cl_g_history)

temperature = 307e3            # temperature (m K)
R = 8.315e3                    # Gas Constant (m J/(K mol))
F = 96500e3                    # Faraday's constant (mC/ mol)

E_Cl_g = - R * temperature / F * np.log(Cl_e_history/Cl_g_history)

n_history = np.array(n_history)
m_history = np.array(m_history)
h_history = np.array(h_history)

print("phi_M_n_init =", potential_history_n[-1, 2])
print("phi_M_g_init =", potential_history_g[-1, 2])
print("K_e_init =", K_e_history[-1, 2])
print("K_n_init =", K_n_history[-1, 2])
print("K_g_init =", K_g_history[-1, 2])
print("Na_e_init =", Na_e_history[-1, 2])
print("Na_n_init =", Na_n_history[-1, 2])
print("Na_g_init =", Na_g_history[-1, 2])
print("Cl_e_init =", Cl_e_history[-1, 2])
print("Cl_n_init =", Cl_n_history[-1, 2])
print("Cl_g_init =", Cl_g_history[-1, 2])

print("n_init =", n_history[-1, 2])
print("m_init =", m_history[-1, 2])
print("h_init =", h_history[-1, 2])

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(potential_history_n[:, 2])
ax[1].plot(potential_history_g[:, 2])
ax[2].plot(K_e_history[:, 2])
#ax[2].plot(K_i_history[:, 2])
ax[3].plot(Na_e_history[:, 2])
#ax[4].plot(Na_i_history[:, 2])
ax[0].set_title("V_n")
ax[1].set_title("V_g")
ax[2].set_title("K_e")
#ax[2].set_title("K_i")
ax[3].set_title("Na_e")
#ax[4].set_title("Na_i")
plt.tight_layout()
fig.savefig("results/figures/ode.png")
#plt.show()
plt.close()

fig = plt.figure()
plt.plot(potential_history_n[:, 2])
#plt.ylim([-72, -70])
plt.tight_layout()
fig.savefig("results/figures/phiM_n.png")

fig = plt.figure()
plt.plot(potential_history_g[:, 2])
#plt.ylim([-72, -70])
plt.tight_layout()
fig.savefig("results/figures/phiM_g.png")

fig = plt.figure()
plt.plot(K_e_history[:, 2])
#plt.ylim([4, 4.1])
plt.tight_layout()
fig.savefig("results/figures/K_e.png")

fig = plt.figure()
plt.plot(Na_e_history[:, 2])
#plt.ylim([99, 100])
plt.tight_layout()
fig.savefig("results/figures/Na_e.png")

fig = plt.figure()
plt.plot(Cl_e_history[:, 2])
#plt.ylim([99, 100])
plt.tight_layout()
fig.savefig("results/figures/Cl_e.png")

fig = plt.figure()
plt.plot(E_Cl_g[:, 2])
plt.plot(potential_history_g[:, 2], linestyle="dotted")
#plt.ylim([99, 100])
plt.tight_layout()
fig.savefig("results/figures/E_Cl_g.png")

# TODO:
# - consider a test where we have dy/dt = A(x)y with y(t=0) = y0
# - after stepping u should be fine
# - add forcing:  dy/dt = A(x)y + f(t) with y(t=0) = y0
# - things are currently quite slow -> multiprocessing?
# - rely on cbc.beat?
