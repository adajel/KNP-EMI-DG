# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original" model

import numpy as np
import math

def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    phi_M_init = -0.0677379636231   # membrane potential (V)

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
    # Param values (values are taken from PDE solver)
    init_values = np.array([0, 0, 0, 0, 0, 0], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("g_leak_Na", 0), ("g_leak_K", 1), \
                      ("E_Na", 2), ("E_K", 3), \
                      ("Cm", 4), ("stim_amplitude", 5)])

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
                       ("Cm", 4), ("stim_amplitude", 5)])

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
    i_Stim = parameters[5] * np.exp(-np.mod(t, 0.02)/0.002)

    # Expressions for the Sodium channel component
    i_Na = (parameters[0] + i_Stim) * \
           (states[0] - parameters[2])

    # Expressions for the Potassium channel component
    i_K = parameters[1] * (states[0] - parameters[3])

    values[0] = (- i_K - i_Na)/parameters[4]
