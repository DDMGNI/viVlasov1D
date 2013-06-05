
import numpy as np


def density(x, L):
    density = 1. + 0.01 * np.cos(0.5 * (x - 0.5 * L))

    return density
