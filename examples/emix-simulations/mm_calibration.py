# We here define an extended membrane ODE system where we have:
#
# 1) Hodgkin Huxley ODES + leak for neuron (as defined in mm_hh)
# 2) Glial mechanism (as defined mm_glial.py)
# 3) ODEs to keep track of concentrations
#
# The system is solved until the system reached a steady state which is used to
# set the initial condition for the membrane variables and concentration in the
# full KNP-EMI (PDE/ODE) system.

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """

    # Init values
    phi_M_n_init = -74.38
    phi_M_g_init = -83.08
    K_e_init = 3.32
    K_n_init = 124.15
    K_g_init = 102.75
    Na_e_init = 100.71
    Na_n_init = 12.83
    Na_g_init = 12.39
    n_init = 0.18
    m_init = 0.01
    h_init = 0.85

    init_values = np.array([m_init, h_init, n_init, phi_M_n_init, phi_M_g_init, \
                            K_e_init, K_n_init, K_g_init, \
                            Na_e_init, Na_n_init, Na_g_init], dtype=np.float_)

    # State indices and limit checker
    state_inds = dict([("m", 0), ("h", 1), ("n", 2), ("V_n", 3), ("V_g", 4),
                      ("K_e", 5), ("K_n", 6), ("K_g", 7),
                      ("Na_e", 8), ("Na_n", 9), ("Na_g", 10)])

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

    # Membrane parameters (as defined in mm_hh.py and mm_glial.py)
    g_Na_bar = 120         # Na max conductivity (mS/cm**2)
    g_K_bar = 36           # K max conductivity (mS/cm**2)
    g_leak_Na_n = 0.1      # Na leak conductivity (mS/cm**2)
    g_leak_K_n  = 0.4      # K leak conductivity (mS/cm**2)

    g_leak_Na_g = 0.1      # Na leak conductivity (mS/cm**2)
    g_leak_K_g  = 1.7      # K leak conductivity (mS/cm**2)
    I_max_g = 50           # max pump strength (muA/cm^2)

    m_K = 2                # threshold ECS K (mol/m^3)
    m_Na = 7.7             # threshold ICS Na (mol/m^3)
    I_max_n = 44.9         # max pump strength (muA/cm^2)
    C_M = 2.0              # Faraday's constant (mC/ mol)

    # Set initial parameter values
    init_values = np.array([g_Na_bar, g_K_bar, \
                            g_leak_Na_n, g_leak_K_n, \
                            g_leak_Na_g, g_leak_K_g, \
                            C_M, 0, \
                            m_K, m_Na, I_max_n, I_max_g], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("g_Na_bar", 0), ("g_K_bar", 1),
                      ("g_leak_Na_n", 2), ("g_leak_K_n", 3),
                      ("g_leak_Na_g", 4), ("g_leak_K_g", 5),
                      ("Cm", 6), ("stim_amplitude", 7),
                      ("m_K", 8), ("m_Na", 9), ("I_max_n", 10),
                      ("I_max_g", 11)])

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
    # State indices and limit checker
    state_inds = dict([("m", 0), ("h", 1), ("n", 2), ("V_n", 3), ("V_g", 4),
                       ("K_e", 5), ("K_n", 6), ("K_g", 7),
                       ("Na_e", 8), ("Na_n", 9), ("Na_g", 10)])

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
                       ("g_leak_Na_n", 2), ("g_leak_K_n", 3),
                       ("g_leak_Na_g", 4), ("g_leak_K_g", 5),
                       ("Cm", 6), ("stim_amplitude", 7),
                       ("m_K", 8), ("m_Na", 9), ("I_max_n", 10),
                       ("I_max_g", 11)])

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

    # Physical parameters (PDEs)
    temperature = 300e3            # temperature (m K)
    R = 8.314e3                    # Gas Constant (m J/(K mol))
    F = 96485e3                    # Faraday's constant (mC/ mol)

    ICS_vol = 3.42e-11/2.0         # ICS volume (cm^3)
    ECS_vol = 7.08e-11             # ECS volume (cm^3)
    surface = 2.29e-6              # membrane surface (cm²)

    K_g_init = 102.74050220804774
    K_e_init = 3.32597273958481

    K_e = states[5]
    K_n = states[6]
    K_g = states[7]

    Na_e = states[8]
    Na_n = states[9]
    Na_g = states[10]

    E_Na_n = R * temperature / F * np.log(Na_e/Na_n) #4
    E_K_n = R * temperature / F * np.log(K_e/K_n)  #5

    E_Na_g = R * temperature / F * np.log(Na_e/Na_g) #4
    E_K_g = R * temperature / F * np.log(K_e/K_g)  #5
    E_K_init = R * temperature / F * np.log(K_e_init/K_g_init)  #5

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
    i_Stim = parameters[7] * np.exp(-np.mod(t, 20.0)/2.0)

    i_pump_n = parameters[10] / ((1 + parameters[8] / K_e) ** 2 \
           * (1 + parameters[9] / Na_n) ** 3)

    i_pump_g = parameters[11] / ((1 + parameters[8] / K_e) ** 2 \
               * (1 + parameters[9] / Na_g) ** 3)

    # set conductance
    dphi = states[4] - E_K_g
    A = 1 + np.exp(18.4/42.4)                                  # shorthand
    B = 1 + np.exp(-(0.1186e3 + E_K_init)/0.0441e3)            # shorthand
    C = 1 + np.exp((dphi + 0.0185e3)/0.0425e3)                 # shorthand
    D = 1 + np.exp(-(0.1186e3 + states[4])/0.0441e3)           # shorthand
    g_Kir = np.sqrt(K_e/K_e_init)*(A*B)/(C*D)

    # define and return current
    I_Kir = parameters[5]*g_Kir*(states[4] - E_K_g)          # umol/(cm^2*ms)

    # Expressions for the Sodium channel component
    i_Na_n = (parameters[2] + parameters[0]*states[1]*math.pow(states[0], 3) + i_Stim) * \
             (states[3] - E_Na_n) + 3 * i_pump_n

    # Expressions for the Potassium channel component
    i_K_n = (parameters[3] + parameters[1]*math.pow(states[2], 4)) * \
            (states[3] - E_K_n) - 2 * i_pump_n

    # Expressions for the Sodium channel component
    i_Na_g = parameters[4] * (states[4] - E_Na_g) + 3 * i_pump_g

    # Expressions for the Potassium channel component
    i_K_g = I_Kir - 2 * i_pump_g

    # Expression for phi_M_n
    values[3] = (- i_K_n - i_Na_n)/parameters[6]

    # Expression for phi_M_g
    values[4] = (- i_K_g - i_Na_g)/parameters[6]

    ### Extension to calculate ECS and ICS ion concentrations ###

    # Expression for K_e
    values[5] = i_K_n * surface / (F * ECS_vol) \
              + i_K_g * surface / (F * ECS_vol)

    # Expression for K_n
    values[6] = - i_K_n  * surface / (F * ICS_vol)

    # Expression for K_g
    values[7] = - i_K_g  * surface / (F * ICS_vol)

    # Expression for Na_e
    values[8] = i_Na_n  * surface / (F * ECS_vol) \
              + i_Na_g  * surface / (F * ECS_vol)

    # Expression for Na_n
    values[9] = - i_Na_n * surface / (F * ICS_vol)

    # Expression for Na_g
    values[10] = - i_Na_g * surface / (F * ICS_vol)
