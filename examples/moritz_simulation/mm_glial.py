# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original" model

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    phi_M_init = -85.85765274084892

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
    g_leak_Na = 0.1         # Na leak conductivity (mS/cm**2)
    g_leak_K  = 1.696       # K leak conductivity (mS/cm**2)
    g_leak_Cl = 0.05        # Cl leak conductivity (mS/cm**2)

    m_K = 1.5               # threshold ECS K (mol/m^3)
    m_Na = 10               # threshold ICS Na (mol/m^3)
    I_max = 10.75975        # max pump strength (muA/cm^2)

    K_e_init = 3.092970607490389
    K_i_init = 99.3100014897692

    # Set initial parameter values
    init_values = np.array([g_leak_Cl, g_leak_Na, g_leak_K, \
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            m_K, m_Na, I_max, K_e_init, K_i_init], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("g_leak_Cl", 0), ("g_leak_Na", 1), ("g_leak_K", 2), \
                      ("E_Cl", 3), ("E_Na", 4), ("E_K", 5), \
                      ("Cm", 6), ("stim_amplitude", 7), \
                      ("I_ch_Na", 8), ("I_ch_K", 9), ("I_ch_Cl", 10), \
                      ("K_e", 11), ("Na_i", 12), \
                      ("m_K", 13), ("m_Na", 14), ("I_max", 15), \
                      ("K_e_init", 16), ("K_i_init", 17)])

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
    param_inds = dict([("g_leak_Cl", 0), ("g_leak_Na", 1), ("g_leak_K", 2), \
                       ("E_Cl", 3), ("E_Na", 4), ("E_K", 5), \
                       ("Cm", 6), ("stim_amplitude", 7), \
                       ("I_ch_Na", 8), ("I_ch_K", 9), ("I_ch_Cl", 10), \
                       ("K_e", 11), ("Na_i", 12), \
                       ("m_K", 13), ("m_Na", 14), ("I_max", 15), \
                       ("K_e_init", 16), ("K_i_init", 17)])

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
    #

    g_leak_Cl = parameters[0]
    g_leak_Na = parameters[1]
    g_leak_K = parameters[2]
    E_Cl = parameters[3]
    E_Na = parameters[4]
    E_K = parameters[5]
    Cm = parameters[6]
    stim_amplitude = parameters[7]
    I_ch_Na = parameters[8]
    I_ch_K = parameters[9]
    I_ch_Cl = parameters[10]
    K_e = parameters[11]
    Na_i = parameters [12]
    m_K = parameters[13]
    m_Na = parameters[14]
    I_max = parameters[15]
    K_e_init = parameters[16]
    K_i_init = parameters[17]

    # Physical parameters (PDEs)
    temperature = 307e3            # temperature (m K)
    R = 8.315e3                    # Gas Constant (m J/(K mol))
    F = 96500e3                    # Faraday's constant (mC/ mol)

    i_pump = I_max \
           * (K_e / (K_e + m_K)) \
           * (Na_i ** (1.5) / (Na_i ** (1.5) + m_Na ** (1.5)))

    # set conductance
    E_K_init = R * temperature / F * np.log(K_e_init/K_i_init)
    dphi = states[0] - E_K
    A = 1 + np.exp(18.4/42.4)                                  # shorthand
    B = 1 + np.exp(-(0.1186e3 + E_K_init)/0.0441e3)            # shorthand
    C = 1 + np.exp((dphi + 0.0185e3)/0.0425e3)                 # shorthand
    D = 1 + np.exp(-(0.1186e3 + states[0])/0.0441e3)           # shorthand
    g_Kir = np.sqrt(K_e/K_e_init)*(A*B)/(C*D)

    # define and return current
    i_Kir = g_leak_K * g_Kir * (states[0] - E_K)              # umol/(cm^2*ms)

    # Expressions for the Sodium channel component
    i_Na = g_leak_Na * (states[0] - E_Na) + 3 * i_pump

    # Expressions for the Potassium channel component
    i_K = i_Kir - 2 * i_pump

    # Expressions for the Chloride channel component
    i_Cl = g_leak_Cl * (states[0] - E_Cl)

    # set I_ch_Na
    parameters[8] = i_Na
    # set I_ch_K
    parameters[9] = i_K
    # set I_ch_Cl
    parameters[10] = i_Cl

    # update membrane potential
    values[0] = (- i_K - i_Na - i_Cl)/Cm
