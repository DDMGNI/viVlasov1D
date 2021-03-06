
import numpy as np


def temperature(x):
    pass


def density(x, L):
#    weak Jeans instability
#    density = 1. + 0.01 * np.cos(0.8 * (x - 0.5 * L))

#    strong Jeans instability
#    density = 1. + 0.01 * np.cos(0.1 * (x - 0.5 * L))

#    linear Landau damping
#    return 1. + 0.01 * np.cos(0.5 * (x - 0.5 * L))
    
#    nonlinear Landau damping
    density = 1. + (np.cos(0.5 * x))
    
    return density



def distribution(x, y):
    
#     if y <= 2. and y >= -2.:
#         f = 1.
#     else:
#         f = 0.
    
    
    f = 1. + 0.5 * (np.cos(x + y*np.pi) - np.cos(x - y*np.pi))
    
    return f

