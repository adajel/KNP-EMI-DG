import os
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import mm_hh_calibration_ODE as ode
#import mm_glial_calibration_ODE as ode
from knpemidg.membrane import MembraneModel
from collections import namedtuple

mesh = df.UnitSquareMesh(2, 2)
V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)

facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
tag = 0

#stimulus = None

g_syn_bar = 5
#g_syn_bar = 0
stimulus = {'stim_amplitude': g_syn_bar}

membrane = MembraneModel(ode, facet_f=facet_f, tag=tag, V=V)

V_index = ode.state_indices('V')
K_e_index = ode.state_indices('K_e')
K_i_index = ode.state_indices('K_i')
Na_e_index = ode.state_indices('Na_e')
Na_i_index = ode.state_indices('Na_i')

#n_index = ode.state_indices('n')
#m_index = ode.state_indices('m')
#h_index = ode.state_indices('h')

potential_history = []
K_e_history = []
K_i_history = []
Na_e_history = []
Na_i_history = []

#n_history = []
#m_history = []
#h_history = []

#for _ in range(50000):
for _ in range(500):
#for _ in range(50000):
    membrane.step_lsoda(dt=0.1, stimulus=stimulus)

    potential_history.append(1*membrane.states[:, V_index])
    K_e_history.append(1*membrane.states[:, K_e_index])
    K_i_history.append(1*membrane.states[:, K_i_index])
    Na_e_history.append(1*membrane.states[:, Na_e_index])
    Na_i_history.append(1*membrane.states[:, Na_i_index])

    #n_history.append(1*membrane.states[:, n_index])
    #m_history.append(1*membrane.states[:, m_index])
    #h_history.append(1*membrane.states[:, h_index])

potential_history = np.array(potential_history)
K_e_history = np.array(K_e_history)
K_i_history = np.array(K_i_history)
Na_e_history = np.array(Na_e_history)
Na_i_history = np.array(Na_i_history)

#n_history = np.array(n_history)
#m_history = np.array(m_history)
#h_history = np.array(h_history)

print("V", potential_history[-1, 2])
print("K_e", K_e_history[-1, 2])
print("K_i", K_i_history[-1, 2])
print("Na_e", Na_e_history[-1, 2])
print("Na_i", Na_i_history[-1, 2])

#print("n", n_history[-1, 2])
#print("m", m_history[-1, 2])
#print("h", h_history[-1, 2])

fig, ax = plt.subplots(5, 1, sharex=True)
ax[0].plot(potential_history[:, 2])
ax[1].plot(K_e_history[:, 2])
ax[2].plot(K_i_history[:, 2])
ax[3].plot(Na_e_history[:, 2])
ax[4].plot(Na_i_history[:, 2])
ax[0].set_title("V")
ax[1].set_title("K_e")
ax[2].set_title("K_i")
ax[3].set_title("Na_e")
ax[4].set_title("Na_i")
plt.tight_layout()
fig.savefig("ode.png")
#plt.show()
plt.close()

fig = plt.figure()
plt.plot(potential_history[:, 2])
#plt.ylim([-72, -70])
plt.tight_layout()
fig.savefig("phiM.png")

fig = plt.figure()
plt.plot(K_e_history[:, 2])
#plt.ylim([4, 4.1])
plt.tight_layout()
fig.savefig("K_e.png")

fig = plt.figure()
plt.plot(Na_e_history[:, 2])
#plt.ylim([99, 100])
plt.tight_layout()
fig.savefig("Na_e.png")

# TODO:
# - consider a test where we have dy/dt = A(x)y with y(t=0) = y0
# - after stepping u should be fine
# - add forcing:  dy/dt = A(x)y + f(t) with y(t=0) = y0
# - things are currently quite slow -> multiprocessing?
# - rely on cbc.beat?
