
import numpy as np

from data import maxwellian_grid


def distribution(grid):
    temperature = np.ones(grid.nx)
    density = 1. + 0.05 * np.cos(0.5 * grid.xGrid)
    
    f = maxwellian_grid(grid, temperature)
    
    for iv in range(0, grid.nv):
        f[:,iv] *= (grid.vGrid[iv])**2
    
    if grid.nv % 2 != 0:
        f[:,grid.nv/2] = f[:,1]
    
    if grid.is_dirichlet():
        f[:, 0] = 0.
        f[:,-1] = 0.
        
    for ix in range(0, grid.nx):
        f[ix,:] /= f[ix].sum() * grid.hv
    
    for ix in range(0, grid.nx):
        f[ix,:] *= density[ix]
    
#    for ix in range(0, grid.nx):
#        for iv in range(0, grid.nv):
#            if f[ix,iv] <= 0.0:
#                print(ix, iv, f[ix,iv])
    
    return f 

