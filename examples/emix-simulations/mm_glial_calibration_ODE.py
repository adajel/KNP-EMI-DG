# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original" model

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """

    # Init values
    phi_M_init = -81.73542331243227

    # Initial values
    #Na_i_init = 15.189                          # Intracellular Na concentration
    #Na_e_init = 99.959                          # extracellular Na concentration
    #K_i_init  = 99.959                          # intracellular K concentration
    #K_e_init  = 4.0900                          # extracellular K concentration

    K_i_init = 101.45093275680335
    Na_i_init = 13.701595802750164
    K_e_init = 3.369320617474813
    Na_e_init = 100.67749185800223

    init_values = np.array([phi_M_init, K_e_init, K_i_init, \
                            Na_e_init, Na_i_init], dtype=np.float_)

    # State indices and limit checker
    state_ind = dict([("V", 0), ("K_e", 1), ("K_i", 2), ("Na_e", 3), ("Na_i", 4)])

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

    K_i_init  = 99.959                          # intracellular K concentration
    K_e_init  = 4.0900                          # extracellular K concentration

    # Membrane parameters
    g_Na_bar = 0           # Na max conductivity (mS/cm**2)
    g_K_bar = 0            # K max conductivity (mS/cm**2)
    g_leak_Na = 0.1        # Na leak conductivity (mS/cm**2)
    g_leak_K  = 1.7        # K leak conductivity (mS/cm**2)

    m_K = 2.0              # threshold ECS K (mol/m^3) - yao 2011
    m_Na = 7.7             # threshold ICS Na (mol/m^3) - yao 2011
    I_max = 43             # max pump strength (muA/cm^2)
    C_M = 2.0              # Faraday's constant (mC/ mol)

    # Set initial parameter values
    init_values = np.array([g_Na_bar, g_K_bar, \
                            g_leak_Na, g_leak_K, \
                            0, 0, C_M, 0, \
                            0, 0, 0, 0, 0,
                            m_K, m_Na, I_max, K_e_init, K_i_init], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("g_Na_bar", 0), ("g_K_bar", 1), \
                      ("g_leak_Na", 2), ("g_leak_K", 3), \
                      ("E_Na", 4), ("E_K", 5), \
                      ("Cm", 6), ("stim_amplitude", 7), \
                      ("I_ch_Na", 8), ("I_ch_K", 9), ("I_ch_Cl", 10), \
                      ("K_e", 11), ("Na_i", 12), \
                      ("m_K", 13), ("m_Na", 14), ("I_max", 15),
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
    state_inds = dict([("V", 0), ("K_e", 1), ("K_i", 2), ("Na_e", 3), ("Na_i", 4)])

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
                      ("Na_i", 12), ("m_K", 13), ("m_Na", 14), ("I_max", 15),\
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

    # Physical parameters (PDEs)
    temperature = 300e3            # temperature (m K)
    R = 8.314e3                    # Gas Constant (m J/(K mol))
    F = 96485e3                    # Faraday's constant (mC/ mol)

    ICS_vol = 3.42e-11             # ICS volume (cm^3)
    ECS_vol = 7.08e-11             # ECS volume (cm^3)
    surface = 2.29e-6              # membrane surface (cm²)

    K_e = states[1] #11
    K_i = states[2]
    Na_e = states[3]
    Na_i = states[4] #12

    K_e_init = parameters[16]
    K_i_init = parameters[17]

    E_Na = R * temperature / F * np.log(Na_e/Na_i) #4
    E_K = R * temperature / F * np.log(K_e/K_i)  #5
    E_K_init = R * temperature / F * np.log(K_e_init/K_i_init)  #5

    i_pump = parameters[15] / ((1 + parameters[13] / K_e) ** 2 \
           * (1 + parameters[14] / Na_i) ** 3)

    # set conductance
    dphi = states[0] - E_K
    A = 1 + np.exp(18.4/42.4)                                  # shorthand
    B = 1 + np.exp(-(0.1186e3 + E_K_init)/0.0441e3)            # shorthand
    C = 1 + np.exp((dphi + 0.0185e3)/0.0425e3)                 # shorthand
    D = 1 + np.exp(-(0.1186e3 + states[0])/0.0441e3)           # shorthand
    g_Kir = np.sqrt(K_e/K_e_init)*(A*B)/(C*D)

    # define and return current
    I_Kir = parameters[3]*g_Kir*(states[0] - E_K)           # umol/(cm^2*ms)

    # Expressions for the Sodium channel component
    i_Na = parameters[2] * (states[0] - E_Na) + 3 * i_pump

    # Expressions for the Potassium channel component
    i_K = I_Kir - 2 * i_pump

    # set I_ch_Na
    parameters[8] = i_Na
    # set I_ch_K
    parameters[9] = i_K
    # set I_ch_Cl
    parameters[10] = 0.0

    # Expression for phi_M
    values[0] = (- i_K - i_Na)/parameters[6]

    # Expression for K_e
    values[1] = i_K * surface / (F * ECS_vol)

    # Expression for K_i
    values[2] = - i_K  * surface / (F * ICS_vol)

    # Expression for Na_e
    values[3] = i_Na  * surface / (F * ECS_vol)

    # Expression for Na_i
    values[4] = - i_Na * surface / (F * ICS_vol)
