# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original" model

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    n_init = 0.18821645700362638
    m_init = 0.016651023270342777
    h_init = 0.8541791472445746
    phi_M_init = -74.3848784437955

    init_values = np.array([m_init, h_init, n_init, phi_M_init], dtype=np.float_)

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
    g_Na_bar = 120          # Na max conductivity (mS/cm**2)
    g_K_bar = 36            # K max conductivity (mS/cm**2)
    g_leak_Na = 0.1         # Na leak conductivity (mS/cm**2)
    g_leak_K  = 0.4         # K leak conductivity (mS/cm**2)

    m_K = 2                 # threshold ECS K (mol/m^3)
    m_Na = 7.7              # threshold ICS Na (mol/m^3)
    I_max = 44.9            # max pump strength (muA/cm^2)

    # Set initial parameter values
    init_values = np.array([g_Na_bar, g_K_bar, \
                            g_leak_Na, g_leak_K, \
                            0, 0, 0, 0, \
                            0, 0, 0, 0, 0,
                            m_K, m_Na, I_max, 0], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("g_Na_bar", 0), ("g_K_bar", 1), \
                      ("g_leak_Na", 2), ("g_leak_K", 3), \
                      ("E_Na", 4), ("E_K", 5), \
                      ("Cm", 6), ("stim_amplitude", 7), \
                      ("I_ch_Na", 8), ("I_ch_K", 9), ("I_ch_Cl", 10), \
                      ("K_e", 11), ("Na_i", 12), \
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
    param_inds = dict([("g_Na_bar", 0), ("g_K_bar", 1), \
                      ("g_leak_Na", 2), ("g_leak_K", 3), \
                      ("E_Na", 4), ("E_K", 5), ("Cm", 6), \
                      ("stim_amplitude", 7), ("I_ch_Na", 8), \
                      ("I_ch_K", 9), ("I_ch_Cl", 10), ("K_e", 11), \
                      ("Na_i", 12), ("m_K", 13), ("m_Na", 14), 
                      ("I_max", 15), ("E_Cl", 16)])

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

    alpha_m = 0.1 * (states[3] + 40.0)/(1.0 - math.exp(-(states[3] + 40.0) / 10.0))
    beta_m = 4.0 * math.exp(-(states[3] + 65.0) / 18.0)

    alpha_h = 0.07 * math.exp(-(states[3] + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + math.exp(-(states[3] + 35.0) / 10.0))

    alpha_n = 0.01 * (states[3] + 55.0)/(1.0 - math.exp(-(states[3] + 55.0) / 10.0))
    beta_n = 0.125 * math.exp(-(states[3] + 65) / 80.0)

    # Expressions for the m gate component
    values[0] = (1 - states[0])*alpha_m - states[0]*beta_m

    # Expressions for the h gate component
    values[1] = (1 - states[1])*alpha_h - states[1]*beta_h

    # Expressions for the n gate component
    values[2] = (1 - states[2])*alpha_n - states[2]*beta_n

    # Expressions for the Membrane component
    i_Stim = parameters[7] * np.exp(-np.mod(t, 20.0)/2.0)#*(t < 45)

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
