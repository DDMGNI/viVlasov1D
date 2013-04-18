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
        
        
        self.EJ1       = 0.0         # current total energy
        self.EJ1_kin   = 0.0         # current kinetic energy
        self.EJ1_pot   = 0.0         # current potential energy
        self.EJ1_error = 0.0         # error in total energy (E-E0)/E0
        
        self.EJ2       = 0.0         # current total energy
        self.EJ2_kin   = 0.0         # current kinetic energy
        self.EJ2_pot   = 0.0         # current potential energy
        self.EJ2_error = 0.0         # error in total energy (E-E0)/E0
        
        self.EJ4       = 0.0         # current total energy
        self.EJ4_kin   = 0.0         # current kinetic energy
        self.EJ4_pot   = 0.0         # current potential energy
        self.EJ4_error = 0.0         # error in total energy (E-E0)/E0
        
        self.EJ1_0     = 0.0         # initial total energy
        self.EJ2_0     = 0.0         # initial total energy
        self.EJ4_0     = 0.0         # initial total energy
        
        self.P0        = 0.0         # initial total momentum
        self.P         = 0.0         # current total momentum
        self.P_error   = 0.0         # error in total momentum (P-P0)/P0
        
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
        
        
        # J1
        Ekin = 0.0
        Epot = 0.0
        
        if f != None:
            for ix in np.arange(0, nx):
                ixp = (ix+1) % nx
                
                for iv in np.arange(0, nv-1):
                    
                    Ekin += ( 
                            + f[ix,  iv  ]
                            + f[ixp, iv  ]
                            + f[ixp, iv+1]
                            + f[ix,  iv+1]
                          ) * ( 
                            + h0[ix,  iv  ]
                            + h0[ixp, iv  ]
                            + h0[ixp, iv+1]
                            + h0[ix,  iv+1]
                            )
                    
                    Epot += ( 
                            + f[ix,  iv  ]
                            + f[ixp, iv  ]
                            + f[ixp, iv+1]
                            + f[ix,  iv+1]
                          ) * ( 
                            + h1[ix,  iv  ]
                            + h1[ixp, iv  ]
                            + h1[ixp, iv+1]
                            + h1[ix,  iv+1]
                            )
        
                    Epot += ( 
                            + f[ix,  iv  ]
                            + f[ixp, iv  ]
                            + f[ixp, iv+1]
                            + f[ix,  iv+1]
                          ) * ( 
                            + h2[ix,  iv  ]
                            + h2[ixp, iv  ]
                            + h2[ixp, iv+1]
                            + h2[ix,  iv+1]
                            )
        
        self.EJ1_kin = Ekin * self.grid.hx * self.grid.hv * 0.25 * 0.25
        self.EJ1_pot = Epot * self.grid.hx * self.grid.hv * 0.25 * 0.25
        
        
        # J2
        Ekin = 0.0
        Epot = 0.0
        
        if f != None:
            for ix in np.arange(0, nx):
                ixm = (ix-1+nx) % nx
                ixp = (ix+1+nx) % nx
                
                for iv in np.arange(1, nv-1):
                    
                    Ekin += ( 
                              + f[ixm, iv  ]
                              + f[ixp, iv  ]
                              + f[ix,  iv-1]
                              + f[ix,  iv+1]
                            ) * (
                              + h0[ixm, iv  ]
                              + h0[ixp, iv  ]
                              + h0[ix,  iv-1]
                              + h0[ix,  iv+1]
                            )
                    
                    Epot += ( 
                              + f[ixm, iv  ]
                              + f[ixp, iv  ]
                              + f[ix,  iv-1]
                              + f[ix,  iv+1]
                            ) * ( 
                              + h1[ixm, iv  ]
                              + h1[ixp, iv  ]
                              + h1[ix,  iv-1]
                              + h1[ix,  iv+1]
                            )
        
                    Epot += ( 
                              + f[ixm, iv  ]
                              + f[ixp, iv  ]
                              + f[ix,  iv-1]
                              + f[ix,  iv+1]
                            ) * (
                              + h2[ixm, iv  ]
                              + h2[ixp, iv  ]
                              + h2[ix,  iv-1]
                              + h2[ix,  iv+1]
                            )
        
        self.EJ2_kin = Ekin * self.grid.hx * self.grid.hv * 0.25 * 0.25
        self.EJ2_pot = Epot * self.grid.hx * self.grid.hv * 0.25 * 0.25
        
        
        # J4
        Ekin = 0.0
        Epot = 0.0
        
        if f != None:
            for ix in np.arange(0, nx):
                ixm = (ix-1+nx) % nx
                ixp = (ix+1+nx) % nx
                
                for iv in np.arange(1, nv-1):
                    
                    Ekin += ( 
                              + f[ixm, iv  ]
                              + f[ixp, iv  ]
                              + 4. * f[ix,  iv  ]
                              + f[ix,  iv-1]
                              + f[ix,  iv+1]
                            ) * (
                              + h0[ixm, iv  ]
                              + h0[ixp, iv  ]
                              + 4. * h0[ix,  iv  ]
                              + h0[ix,  iv-1]
                              + h0[ix,  iv+1]
                            )
                    
                    Epot += ( 
                              + f[ixm, iv  ]
                              + f[ixp, iv  ]
                              + 4. * f[ix,  iv  ]
                              + f[ix,  iv-1]
                              + f[ix,  iv+1]
                            ) * ( 
                              + h1[ixm, iv  ]
                              + h1[ixp, iv  ]
                              + 4. * h1[ix,  iv  ]
                              + h1[ix,  iv-1]
                              + h1[ix,  iv+1]
                            )
        
                    Epot += ( 
                              + f[ixm, iv  ]
                              + f[ixp, iv  ]
                              + 4. * f[ix,  iv  ]
                              + f[ix,  iv-1]
                              + f[ix,  iv+1]
                            ) * (
                              + h2[ixm, iv  ]
                              + h2[ixp, iv  ]
                              + 4. * h2[ix,  iv  ]
                              + h2[ix,  iv-1]
                              + h2[ix,  iv+1]
                            )
        
        self.EJ4_kin = Ekin * self.grid.hx * self.grid.hv * 0.125 * 0.125
        self.EJ4_pot = Epot * self.grid.hx * self.grid.hv * 0.125 * 0.125
        
        
        # total energy
        self.EJ1 = self.EJ1_kin + 0.5 * self.EJ1_pot
        self.EJ2 = self.EJ2_kin + 0.5 * self.EJ2_pot
        self.EJ4 = self.EJ4_kin + 0.5 * self.EJ4_pot
        
    
    def calculate_energy_error(self):
        '''
        Calculates energy error, i.e. relative difference between E and E0.
        '''
        
        if self.EJ1_0 != 0.0:
            self.EJ1_error = (self.EJ1 - self.EJ1_0) / self.EJ1_0
        else:
            self.EJ1_error = 0.0
    
        if self.EJ2_0 != 0.0:
            self.EJ2_error = (self.EJ2 - self.EJ2_0) / self.EJ2_0
        else:
            self.EJ2_error = 0.0
    
        if self.EJ4_0 != 0.0:
            self.EJ4_error = (self.EJ4 - self.EJ4_0) / self.EJ4_0
        else:
            self.EJ4_error = 0.0
    

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
            for ix in np.arange(0, nx):
                ixp = (ix+1) % nx
                
                for iv in np.arange(0, nv-1):
                    
                    self.P += ( 
                            + f[ix,  iv  ]
                            + f[ixp, iv  ]
                            + f[ixp, iv+1]
                            + f[ix,  iv+1]
                          ) * ( 
                            + self.grid.vGrid[iv  ]
                            + self.grid.vGrid[iv+1]
                            )
            
            self.P *= self.mass * 0.25 * 0.5 * self.grid.hx * self.grid.hv
        
    
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
            self.EJ1_0 = self.EJ1
            self.EJ2_0 = self.EJ2
            self.EJ4_0 = self.EJ4
        
            self.P0    = self.P
