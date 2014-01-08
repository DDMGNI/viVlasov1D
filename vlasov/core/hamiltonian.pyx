'''
Created on Mar 20, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import  numpy as np
cimport numpy as np

from libc.math cimport pow


class Hamiltonian(object):
    '''
    Discrete representation of the Hamiltonian of the Vlasov-Poisson system.
    '''


    def __init__(self, grid, mass=1.0, potential=None, hdf5=None):
        '''
        Constructor
        '''
        
        self.grid  = grid       # grid object
        self.mass  = mass       # particle mass
        
        self.f  = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )      # distribution function (data array only),
                                                                                  # needed for calculation of total energy
        self.h  = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )      # total hamiltonian
        
        self.h0 = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )      # kinetic   term
        self.h1 = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )      # potential term
        self.h2 = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )      # external  term
        
        
        self.E0      = 0.0         # initial total energy
        self.E       = 0.0         # current total energy
        self.E_error = 0.0         # error in total energy (E-E0)/E0
        
        self.E_kin0  = 0.0         # initial kinetic energy
        self.E_kin   = 0.0         # current kinetic energy
        
        self.E_pot0  = 0.0         # initial potential energy
        self.E_pot   = 0.0         # current potential energy
        
        self.P0      = 0.0         # initial total momentum
        self.P       = 0.0         # current total momentum
        self.P_error = 0.0         # error in total momentum (P-P0)/P0
        
        self.set_hdf5_file(hdf5)
        self.read_from_hdf5(0)
            
    
    
    def update(self, potential=None):
        self.calculate_total_energy()
        self.calculate_total_momentum()
        self.calculate_energy_error()
        self.calculate_momentum_error()
        
    
    
    def calculate_total_energy(self):
        '''
        Calculates total energy integral w.r.t. the given distribution function.
        '''
        
        cdef np.uint64_t nx = self.grid.nx
        cdef np.uint64_t nv = self.grid.nv
        
        cdef np.uint64_t ix, ixm, ixp, iv
        
        cdef np.float64_t Ekin, Epot
        
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.h0
        cdef np.ndarray[np.float64_t, ndim=2] h1 = self.h1 - self.h1.mean()
        cdef np.ndarray[np.float64_t, ndim=2] h2 = self.h2 - self.h2.mean()
        cdef np.ndarray[np.float64_t, ndim=2] f  = self.f
        
        
        Ekin = 0.0
        Epot = 0.0
        
        if f != None:
            for ix in range(0, nx):
                for iv in range(0, nv):
                    
                    Ekin += f[ix,  iv  ] * h0[ix,  iv  ]
                    Epot += f[ix,  iv  ] * h1[ix,  iv  ]
                    Epot += f[ix,  iv  ] * h2[ix,  iv  ]
        
        self.E_kin = Ekin * self.grid.hx * self.grid.hv
        self.E_pot = Epot * self.grid.hx * self.grid.hv
        
        
        # total energy
        self.E = self.E_kin + 0.5 * self.E_pot
        
    
    def calculate_energy_error(self):
        '''
        Calculates energy error, i.e. relative difference between E and E0.
        '''
        
        if self.E0 != 0.0:
            self.E_error = (self.E - self.E0) / self.E0
        else:
            self.E_error = 0.0
    

    def calculate_total_momentum(self):
        '''
        Calculates total momentum w.r.t. the given distribution function.
        '''
        
        cdef np.uint64_t nx = self.grid.nx
        cdef np.uint64_t nv = self.grid.nv
        
        cdef np.uint64_t ix, ixp, iv
        
        cdef np.ndarray[np.float64_t, ndim=2] f  = self.f
        
        self.P = 0.0
        
        if self.f != None:
            for ix in range(0, nx):
                ixp = (ix+1) % nx
                
                for iv in range(0, nv-1):
                    
                    self.P += f[ix, iv] * self.grid.vGrid[iv]

            self.P *= self.mass * self.grid.hx * self.grid.hv
        
    
    def calculate_momentum_error(self):
        '''
        Calculates momentum error, i.e. relative difference between P and P0.
        '''
        if self.P0 != 0.0:
            self.P_error = (self.P - self.P0) / self.P0
        else:
            self.P_error = 0.0
    

    
    def set_hdf5_file(self, hdf5):
        self.hdf5_f  = hdf5['f']
        self.hdf5_h0 = hdf5['h0']
        self.hdf5_h1 = hdf5['h1']
        
        try:
            self.hdf5_h2 = hdf5['h2']
        except KeyError:
            self.hdf5_h2 = None
    
    
    def read_from_hdf5(self, iTime):
        self.h0[:,:] = self.hdf5_h0[iTime,:,:]
        self.h1[:,:] = self.hdf5_h1[iTime,:,:]
        
        if self.hdf5_h2 != None:
            self.h2[:,:] = self.hdf5_h2[iTime,:,:]
        
        self.f[:,:] = self.hdf5_f[iTime,:,:]
        self.h[:,:] = self.h0 + self.h1 + self.h2
        
        self.calculate_total_energy()
        self.calculate_total_momentum()
        self.calculate_energy_error()
        self.calculate_momentum_error()
        
#         if iTime == 0 or iTime == 1:
        if iTime == 0:
            self.P0 = self.P
            self.E0 = self.E
        
            self.E_kin0 = self.E_kin
            self.E_pot0 = self.E_pot

