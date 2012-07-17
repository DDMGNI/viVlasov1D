'''
Created on Mar 21, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import numpy as np


def gaussian_profile(xGrid, xPos, xSigma):
    profile = np.zeros(len(xGrid))
    hx = xGrid[1] - xGrid[0]
    
    for ix in range(0, len(xGrid)):
        profile[ix] = gaussian(xGrid[ix], xGrid[xPos], hx*xSigma)
    
    return profile


def gaussian(x, xcenter, sigma):
    return 1. / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-xcenter)/sigma)**2 )
    



if __name__ == '__main__':
    xGrid  = np.linspace(0.0, 10.0, 101)
    xPos   = 50
    xSigma = 10
    
    profile = gaussian_profile(xGrid, xPos, xSigma)
    
    np.savetxt('gaussian.dat', profile)
    
