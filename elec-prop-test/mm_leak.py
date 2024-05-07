# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original" model

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    #phi_M_init = -0.06831807655880984
    phi_M_init = -68.31807655880984

    init_values = np.array([phi_M_init], dtype=np.float_)

    # State indices and limit checker
    state_ind = dict([("V", 0)])

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
    #g_leak_Na = 1.0     # Na leak conductivity (S/m**2)
    #g_leak_K  = 4.0     # K leak conductivity (S/m**2)
    #m_K = 4.0           # threshold ECS K (mol/m^3)
    #m_Na = 12.0         # threshold ICS Na (mol/m^3)
    #I_max = 1.3         # max pump strength (A/m^2)

    g_leak_Na = 0.1     # Na leak conductivity (mS/cm**2)
    g_leak_K  = 0.4     # K leak conductivity (mS/cm**2)

    m_K = 4.0            # threshold ECS K (mol/m^3)
    m_Na = 12.0          # threshold ICS Na (mol/m^3)
    I_max = 1.3e-3       # max pump strength (muA/m^2)

    # Set initial parameter values
    init_values = np.array([g_leak_Na, g_leak_K, \
                            0, 0, 0, 0, 0, 0, 0, \
                            m_K, m_Na, I_max, \
                            0, 0], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("g_leak_Na", 0), ("g_leak_K", 1), \
                      ("E_Na", 2), ("E_K", 3), \
                      ("Cm", 4), ("stim_amplitude", 5), \
                      ("I_ch_Na", 6), ("I_ch_K", 7), ("I_ch_Cl", 8), \
                      ("m_K", 9), ("m_Na", 10), ("I_max", 11), \
                      ("K_e", 12), ("Na_i", 13)])

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
    state_inds = dict([("V", 0)])

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
    param_inds = dict([("g_leak_Na", 0), ("g_leak_K", 1), \
                       ("E_Na", 2), ("E_K", 3), \
                       ("Cm", 4), ("stim_amplitude", 5), \
                       ("I_ch_Na", 6), ("I_ch_K", 7), ("I_ch_Cl", 8), \
                       ("m_K", 9), ("m_Na", 10), ("I_max", 11), \
                       ("K_e", 12), ("Na_i", 13)])

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
    #assert(len(parameters)) == 10

    # Init return args
    #if values is None:
         #values = np.zeros((4,), dtype=np.float_)
    #else:
         #assert isinstance(values, np.ndarray) and values.shape == (4,)

    # Expressions for stimuli
    i_Stim = parameters[5]

    I_pump = parameters[11] / ((1 + parameters[9] / parameters[12]) ** 2 \
           * (1 + parameters[10] / parameters[13]) ** 3)

    # Expressions for the Sodium channel component
    i_Na = (parameters[0] + i_Stim) * (states[0] - parameters[2]) + 3 * I_pump

    # Expressions for the Potassium channel component
    i_K = parameters[1] * (states[0] - parameters[3]) - 2 * I_pump

    # set I_ch_Na
    parameters[6] = i_Na
    # set I_ch_K
    parameters[7] = i_K
    # set I_ch_Cl
    parameters[8] = 0.0

    values[0] = (- i_K - i_Na)/parameters[4]
