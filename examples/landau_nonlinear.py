"""
Nonlinear Landau Damping

   Initial density: :math:`n(x) = 1 + A \, \cos{( k_{x} \, ( x - L_{x}/2)) }`
   with :math:`A = 0.5, k = 0.5`
"""

import numpy as np


def density(x, L):
    density = 1. + 0.5 * np.cos(0.5 * (x - 0.5 * L))

    return density
