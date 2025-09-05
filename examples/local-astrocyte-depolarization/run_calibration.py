import os
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import mm_calibration as ode
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

# ODE plots
fig = plt.figure(figsize=(16,12))
ax = plt.gca()

ax1 = fig.add_subplot(3,4,1)
plt.title(r'ECS Na$^+$')
plt.plot(Na_e_history[:, 2], linewidth=3, color='b')

ax2 = fig.add_subplot(3,4,2)
plt.title(r'ECS K$^+$')
plt.plot(K_e_history[:, 2], linewidth=3, color='b')

ax3 = fig.add_subplot(3,4,3)
plt.title(r'Neuron Na$^+$')
plt.plot(Na_n_history[:, 2],linewidth=3, color='r')

ax4 = fig.add_subplot(3,4,4)
plt.title(r'Neuron K$^+$')
plt.plot(K_n_history[:, 2],linewidth=3, color='r')

ax3 = fig.add_subplot(3,4,5)
plt.title(r'Glia Na$^+$')
plt.plot(Na_g_history[:, 2],linewidth=3, color='r')

ax4 = fig.add_subplot(3,4,6)
plt.title(r'Glia K$^+$')
plt.plot(K_g_history[:, 2],linewidth=3, color='r')

ax5 = fig.add_subplot(3,4,7)
plt.title(r'Membrane potential neuron')
plt.plot(potential_history_n[:, 2], linewidth=3)

ax6 = fig.add_subplot(3,4,8)
plt.title(r'Membrane potential glial')
plt.plot(potential_history_g[:, 2], linewidth=3)

ax7 = fig.add_subplot(3,4,9)
plt.title(r'Gating variable n')
plt.plot(n_history[:, 2], linewidth=3)

ax8 = fig.add_subplot(3,4,10)
plt.title(r'Gating variable m')
plt.ylabel(r'$\phi_M$ (mV)')
plt.plot(m_history[:, 2], linewidth=3)

ax9 = fig.add_subplot(3,4,11)
plt.title(r'Gating variable h')
plt.plot(h_history[:, 2], linewidth=3)

# make pretty
ax.axis('off')
plt.tight_layout()

# save figure to file
plt.savefig('calibration.svg', format='svg')
plt.close()
