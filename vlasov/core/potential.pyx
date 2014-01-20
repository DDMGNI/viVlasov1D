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


    def __init__(self, grid, hdf5, charge=1.0):
        '''
        Constructor
        '''
        
#         assert grid not None
#         assert hdf5 not None
        
        
        self.average_diagnostics = False
        
        
        self.grid  = grid
        
        self.hdf5_phi = None 
        
        self.E0 = 0.0
        self.E  = 0.0
        self.E_error = 0.0
        
        self.charge = charge
        
        self.phi = None
        self.set_hdf5_file(hdf5)
        self.read_from_hdf5(0)
        self.E0 = self.E
            
        
    
    def calculate_energy(self):
        cdef np.uint64_t nx = self.grid.nx
        
        cdef np.uint64_t ix, ixp
        cdef np.float64_t E
        
        cdef np.ndarray[np.float64_t, ndim=1] phi  = self.phi
        cdef np.ndarray[np.float64_t, ndim=1] tphi = phi - phi.mean()
        
        E = 0.0
        
        for ix in range(0, nx):
            ixp = (ix+1) % nx
            
            E += pow(tphi[ixp] - tphi[ix], 2)
        
        self.E = 0.5 * self.charge * E / self.grid.hx
        # / hx**2 for the square of the derivative
        # * hx**1 for the integration
        
    
    def set_hdf5_file(self, hdf5):
        self.hdf5_phi = hdf5['phi_int']
                
        
    def read_from_hdf5(self, iTime):
        self.phi = self.hdf5_phi[iTime,:]
        
        self.calculate_energy()
        
    
