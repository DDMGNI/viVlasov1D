
import numpy as np
from data.gauss import gaussian_profile


def temperature(grid):
    pass


def density(grid):
    xPos   = grid.nx / 2
    xSigma = grid.nx / 10
    
#    weak Jeans instability
#    density = 1. + 0.01 * np.cos(0.8 * (grid.xGrid - 0.5 * grid.L))

#    strong Jeans instability
#    density = 1. + 0.01 * np.cos(0.1 * (grid.xGrid - 0.5 * grid.L))

#    linear Landau damping
#    density = 1. + 0.01 * np.cos(0.5 * (grid.xGrid - 0.5 * grid.L))
    
#    nonlinear Landau damping
    density = 1. + 0.5 * np.cos(0.5 * (grid.xGrid - 0.5 * grid.L))

#    Gaussian density profile
#    density  = gaussian_profile(grid.xGrid, xPos, xSigma)
#    density /= density.mean()
    
    return density



def potential(grid):
    return np.sin(3.*np.pi * grid.xGrid / grid.xMax)


def distribution(grid):
    f = np.zeros((grid.nx, grid.nv))
#    f[:,0] = 0.
#    f[:,1] = 1.
#    f[:,2] = 2.
#    f[:,3] = 1.
#    f[:,4] = 0.
    f[:,0] = 0.
    f[:,1] = 1.
    f[:,2] = 0.
    
    return f

