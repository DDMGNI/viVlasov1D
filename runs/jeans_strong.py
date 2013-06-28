"""
Strong Jeans Instability

   Initial density: :math:`n(x) = 1 + A \, \cos{( k_{x} \, ( x - L_{x}/2)) }`
   with :math:`A = 0.01, k = 0.2`
"""

import numpy as np


def density(x, L):
    return 1. + 0.01 * np.cos(0.1 * (x - 0.5 * L))

