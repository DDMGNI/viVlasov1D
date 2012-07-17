'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from numpy import exp, pi, zeros

from boltzmann import boltzmannian


def maxwellian_grid(grid, temperature, vOffset=0.0):
    m = zeros((grid.nx, grid.nv))
    
    for ix in range(0, grid.nx):
        for iv in range(0, grid.nv):
            m[ix,iv] = maxwellian(temperature[ix], grid.vGrid[iv], vOffset)
    
    return m

    
def maxwellian(temperature, velocity, vOffset=0.0):
    return boltzmannian(temperature, 0.5 * (velocity+vOffset)**2)
