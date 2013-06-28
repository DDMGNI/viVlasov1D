"""
Jeans Damping

   Initial density: :math:`n(x) = 1 + A \, \cos{( k_{x} \, ( x - L_{x}/2)) }`
   with :math:`A = 0.1, k = 2.0`
"""


import numpy as np


def density(x, L):
    return 1. + 0.1 * np.cos(2.0 * (x - 0.5 * L))

