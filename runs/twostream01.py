
import numpy as np

from data import maxwellian_grid


def distribution(grid):
    vOffset = 1.0
    
    temperature = 0.1 * np.ones(grid.nx)
    density = 1. + 0.05 * np.cos(0.5 * grid.xGrid)
    
    f1 = maxwellian_grid(grid, temperature, +vOffset)
    f2 = maxwellian_grid(grid, temperature, -vOffset)
    
    f = f1+f2
    
    for ix in range(0, grid.nx):
        f[ix,:] /= f[ix].sum()
    
    for ix in range(0, grid.nx):
        f[ix,:] *= density[ix]
    
    return f 

