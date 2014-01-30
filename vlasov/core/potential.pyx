'''
Created on Mar 21, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import  numpy as np
cimport numpy as np

from libc.math cimport pow

#from poisson import Poisson


class Potential(object):
    '''
    Discrete representation of the potential in the Vlasov-Poisson system.
    '''


    def __init__(self, grid, hdf5=None):
        '''
        Constructor
        '''
        
        assert grid is not None
        assert hdf5 is not None
        
        
        self.grid     = grid
        self.hdf5_phi = hdf5['phi_int'] 
        self.phi      = None
        
        if 'charge' in hdf5.attrs:
            self.charge = hdf5.attrs['charge']
        else:
            self.charge = -1.
        
        self.E0 = 0.0
        self.E  = 0.0
        self.E_error = 0.0
        
        self.read_from_hdf5(0)
        self.E0 = self.E
            
        
    
    def calculate_energy(self):
        cdef int ix, ixp
        cdef double E = 0.0
        
        cdef double[:] phi = self.phi
        
        for ix in range(0, self.grid.nx):
            ixp = (ix+1) % self.grid.nx
            
            E += pow(phi[ixp] - phi[ix], 2)
        
        self.E = 0.5 * self.charge * E / self.grid.hx
        # / hx**2 for the square of the derivative
        # * hx**1 for the integration
        
    
    def read_from_hdf5(self, iTime):
        self.phi = self.hdf5_phi[iTime,:]
        
        self.calculate_energy()
        
    
