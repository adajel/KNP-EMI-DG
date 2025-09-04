# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original"
# model

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    n_init = 0.1882020248041632         # gating variable n
    m_init = 0.016648440745822956       # gating variable m
    h_init = 0.8542015627820805         # gating variable h
    phi_M_init = -0.07438609374462003   # membrane potential (V)

    init_values = np.array([m_init, h_init, n_init, phi_M_init], dtype=np.float64)

    # State indices and limit checker
    state_ind = dict([("m", 0), ("h", 1), ("n", 2), ("V", 3)])

    for state_name, value in values.items():
        if state_name not in state_ind:
            raise ValueError("{0} is not a state.".format(state_name))
        ind = state_ind[state_name]

        # Assign value
        init_values[ind] = value

    return init_values

def init_parameter_values(**values):
    """
    Initialize parameter values
    """

    # Membrane parameters
    g_Na_bar = 1200         # Na max conductivity (S/m**2)
    g_K_bar = 360           # K max conductivity (S/m**2)
    g_leak_Na = 2.0*0.5     # Na leak conductivity (S/m**2)
    g_leak_K  = 8.0*0.5     # K leak conductivity (S/m**2)

    m_K = 2                 # threshold ECS K (mol/m**3)
    m_Na = 7.7              # threshold ICS Na (mol/m**3)
    I_max = 0.449           # max pump strength (A/m**2)

    # Set initial parameter values
    init_values = np.array([g_Na_bar, g_K_bar, \
                            g_leak_Na, g_leak_K, \
                            0, 0, 0, 0, \
                            0, 0, 0, 0, 0,
                            m_K, m_Na, I_max, 0], dtype=np.float64)


    # Parameter indices and limit checker
    param_ind = dict([("g_Na_bar", 0), ("g_K_bar", 1),
                      ("g_leak_Na", 2), ("g_leak_K", 3),
                      ("E_Na", 4), ("E_K", 5),
                      ("Cm", 6), ("stim_amplitude", 7),
                      ("I_ch_Na", 8), ("I_ch_K", 9), ("I_ch_Cl", 10),
                      ("K_e", 11), ("Na_i", 12),
                      ("m_K", 13), ("m_Na", 14), ("I_max", 15), ("E_Cl", 16)])

    for param_name, value in values.items():
        if param_name not in param_ind:
            raise ValueError("{0} is not a parameter.".format(param_name))
        ind = param_ind[param_name]

        # Assign value
        init_values[ind] = value

    return init_values

def state_indices(*states):
    """
    State indices
    """
    state_inds = dict([("m", 0), ("h", 1), ("n", 2), ("V", 3)])

    indices = []
    for state in states:
        if state not in state_inds:
            raise ValueError("Unknown state: '{0}'".format(state))
        indices.append(state_inds[state])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def parameter_indices(*params):
    """
    Parameter indices
    """
    param_inds = dict([("g_Na_bar", 0), ("g_K_bar", 1),
                      ("g_leak_Na", 2), ("g_leak_K", 3),
                      ("E_Na", 4), ("E_K", 5),
                      ("Cm", 6), ("stim_amplitude", 7),
                      ("I_ch_Na", 8), ("I_ch_K", 9), ("I_ch_Cl", 10),
                      ("K_e", 11), ("Na_i", 12),
                      ("m_K", 13), ("m_Na", 14), ("I_max", 15), ("E_Cl", 16)])

    indices = []
    for param in params:
        if param not in param_inds:
            raise ValueError("Unknown param: '{0}'".format(param))
        indices.append(param_inds[param])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

from numbalsoda import lsoda_sig
from numba import njit, cfunc, jit
import numpy as np
import timeit
import math

@cfunc(lsoda_sig, nopython=True) 
def rhs_numba(t, states, values, parameters):
    """
    Compute the right hand side of the\
        hodgkin_huxley_squid_axon_model_1952_original ODE
    """

    # Expressions for the m gate component
    alpha_m = 0.1e3 * (25. - 1.0e3*(states[3] + 65.0e-3))/(math.exp((25. - 1.0e3*(states[3] + 65.0e-3))/10.) - 1)
    beta_m = 4.e3*math.exp(- 1.0e3*(states[3] + 65.0e-3)/18.)
    values[0] = (1 - states[0])*alpha_m - states[0]*beta_m

    # Expressions for the h gate component
    alpha_h = 0.07e3*math.exp(- 1.0e3*(states[3] + 65.0e-3)/20.)
    beta_h = 1.e3/(math.exp((30.- 1.0e3*(states[3] + 65.0e-3))/10.) + 1)
    values[1] = (1 - states[1])*alpha_h - states[1]*beta_h

    # Expressions for the n gate component
    alpha_n = 0.01e3*(10.- 1.0e3*(states[3] + 65.0e-3))/(math.exp((10.- 1.0e3*(states[3] + 65.0e-3))/10.) - 1.)
    beta_n = 0.125e3*math.exp(- 1.0e3*(states[3] + 65.0e-3) /80.)
    values[2] = (1 - states[2])*alpha_n - states[2]*beta_n

    # Expressions for the Membrane component
    i_Stim = parameters[7] * np.exp(-np.mod(t, 0.03)/0.002)*(t < 125e-3)

    i_pump = parameters[15] / ((1 + parameters[13] / parameters[11]) ** 2 \
           * (1 + parameters[14] / parameters[12]) ** 3)

    # Expressions for the Sodium channel component
    i_Na = (parameters[2] + parameters[0]*states[1]*math.pow(states[0], 3) + i_Stim) * \
           (states[3] - parameters[4]) + 3 * i_pump

    # Expressions for the Potassium channel component
    i_K = (parameters[3] + parameters[1]*math.pow(states[2], 4)) * \
          (states[3] - parameters[5]) - 2 * i_pump

    # set I_ch_Na
    parameters[8] = i_Na
    # set I_ch_K
    parameters[9] = i_K
    # set I_ch_Cl
    parameters[10] = 0.0

    values[3] = (- i_K - i_Na)/parameters[6]
