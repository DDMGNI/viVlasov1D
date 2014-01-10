
import numpy as np


def density(x, L):
    return 1. + 0.01 * np.cos(0.1 * (x - 0.5 * L))

