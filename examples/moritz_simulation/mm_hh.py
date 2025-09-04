# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original" model

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    n_init = 0.17041625484928405
    m_init = 0.01365600905697864
    h_init = 0.8804834256821714
    phi_M_init = -75.93151471235473

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
    g_Na_bar = 120         # Na max conductivity (mS/cm**2)
    g_K_bar = 36           # K max conductivity (mS/cm**2)
    g_leak_Na = 0.1        # Na leak conductivity (mS/cm**2)
    g_leak_K  = 0.4        # K leak conductivity (mS/cm**2)

    m_K = 1.5              # threshold ECS K (mol/m^3)
    m_Na = 10              # threshold ICS Na (mol/m^3)
    I_max = 58.0           # max pump strength (muA/cm^2)

    # Set initial parameter values
    init_values = np.array([g_Na_bar, g_K_bar, \
                            g_leak_Na, g_leak_K, \
                            0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0,
                            m_K, m_Na, I_max], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("g_Na_bar", 0), ("g_K_bar", 1), \
                      ("g_leak_Na", 2), ("g_leak_K", 3), \
                      ("E_Na", 4), ("E_K", 5), ("E_Cl", 6), \
                      ("Cm", 7), ("stim_amplitude", 8), \
                      ("I_ch_Na", 9), ("I_ch_K", 10), ("I_ch_Cl", 11), \
                      ("K_e", 12), ("Na_i", 13), \
                      ("m_K", 14), ("m_Na", 15), ("I_max", 16)])

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
                       ("E_Na", 4), ("E_K", 5), ("E_Cl", 6), \
                       ("Cm", 7), ("stim_amplitude", 8), \
                       ("I_ch_Na", 9), ("I_ch_K", 10), ("I_ch_Cl", 11), \
                       ("K_e", 12), ("Na_i", 13), \
                       ("m_K", 14), ("m_Na", 15), ("I_max", 16)])

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

    # Assign states
    #assert(len(states)) == 4

    # Assign parameters
    #assert(len(parameters)) == 11

    # # Init return args
    # if values is None:
    #     values = np.zeros((4,), dtype=np.float_)
    # else:
    #     assert isinstance(values, np.ndarray) and values.shape == (4,)

    # Parameter indices and limit checker
    g_Na_bar = parameters[0]
    g_K_bar = parameters[1]
    g_leak_Na = parameters[2]
    g_leak_K = parameters[3]
    E_Na = parameters[4]
    E_K = parameters[5]
    E_Cl = parameters[6]
    Cm = parameters[7]
    stim_amplitude = parameters[8]
    I_ch_Na = parameters[9]
    I_ch_K = parameters[10]
    I_ch_Cl = parameters[11]
    K_e = parameters[12]
    Na_i = parameters[13]
    m_K = parameters[14]
    m_Na = parameters[15]
    I_max =  parameters[16]

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

    i_pump = I_max / ((1 + m_K / K_e) ** 2 \
           * (1 + m_Na / Na_i) ** 3)

    # Expressions for the Sodium channel component
    i_Na = (g_leak_Na + g_Na_bar * states[1]*math.pow(states[0], 3)) * \
           (states[3] - E_Na) + 3 * i_pump

    # Expressions for the Potassium channel component
    i_K = (g_leak_K + g_K_bar * math.pow(states[2], 4)) * \
          (states[3] - E_K) - 2 * i_pump

    # set I_ch_Na
    parameters[9] = i_Na
    # set I_ch_K
    parameters[10] = i_K
    # set I_ch_Cl
    parameters[11] = 0.0

    values[3] = (- i_K - i_Na)/Cm
