
import numpy as np


def temperature(x):
    pass


def density(x, L):
    density = 1. + 0.5 * np.cos(0.5 * (x - 0.5 * L))

    return density


def distribution(x, y):
    pass
