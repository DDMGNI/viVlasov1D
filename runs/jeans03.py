
import numpy as np


def density(grid):
    return 1. + 0.1 * np.cos(2.0 * (grid.xGrid - 0.5 * grid.L))
