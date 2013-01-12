'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from numpy import exp, pi, sqrt, zeros


def boltzmannian_grid(grid, temperature, energy):
    b = zeros((grid.nx, grid.nv))
    
    for ix in range(0, grid.nx):
        for iv in range(0, grid.nv):
            b[ix,iv] = boltzmannian(temperature[ix], energy[ix,iv])
    
    return b


def boltzmannian(temperature, energy):
    return sqrt(0.5 / pi) * exp( - energy / temperature )


