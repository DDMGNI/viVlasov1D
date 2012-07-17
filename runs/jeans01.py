
import numpy as np


def density(grid):
    return 1. + 0.01 * np.cos(0.8 * (grid.xGrid - 0.5 * grid.L))

