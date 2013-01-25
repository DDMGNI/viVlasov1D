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


    def __init__(self, grid, mass=1.0, nhist=1, potential=None, distribution=None,
                 hdf5_in=None,  hdf5_out=None, replay=False):
        '''
        Constructor
        '''
        
        self.average_diagnostics = True
        
        
        self.grid  = grid       # grid object
        self.nhist = nhist      # number of timesteps to save in history
        self.mass  = mass       # particle mass
        
        self.f = None           # distribution function (data array only),
                                # needed for calculation of total energy
        
        self.h  = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )      # total hamiltonian
        self.h0 = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )      # kinetic   term
        self.h1 = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )      # potential term
        self.h2 = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )      # external  term
        
        self.E0    = 0.0         # initial total energy
        self.Ekin0 = 0.0         # initial kinetic energy
        self.Epot0 = 0.0         # initial potential energy
        self.E     = 0.0         # current total energy
        self.Ekin  = 0.0         # current kinetic energy
        self.Epot  = 0.0         # current potential energy
        self.E_error = 0.0       # error in total energy (E-E0)/E0
        
        self.P0    = 0.0         # initial total momentum
        self.P     = 0.0         # current total momentum
        self.P_error = 0.0       # error in total momentum (P-P0)/P0
        
        
        if hdf5_in != None and replay:
            self.set_hdf5_file(hdf5_in)
            self.read_from_hdf5(0)
            
            self.Ekin0   = self.Ekin
            self.Epot0   = self.Epot
            self.E0      = self.E
            self.P0      = self.P
            
        else:
            if hdf5_out != None:
                # create HDF5 fields
                self.hdf5_h0 = hdf5_out.create_dataset('h0', (self.grid.nt+1, self.grid.nx, self.grid.nv), '=f8')
                self.hdf5_h1 = hdf5_out.create_dataset('h1', (self.grid.nt+1, self.grid.nx, self.grid.nv), '=f8')
                self.hdf5_h2 = hdf5_out.create_dataset('h2', (self.grid.nt+1, self.grid.nx, self.grid.nv), '=f8')
            
            self.history  = np.zeros( (self.grid.nx, self.grid.nv, self.nhist), dtype=np.float64 )
            self.history0 = np.zeros( (self.grid.nx, self.grid.nv, self.nhist), dtype=np.float64 )
            self.history1 = np.zeros( (self.grid.nx, self.grid.nv, self.nhist), dtype=np.float64 )
            self.history2 = np.zeros( (self.grid.nx, self.grid.nv, self.nhist), dtype=np.float64 )
            
            
            if hdf5_in != None:
                nt = len(hdf5_in['grid']['t'][:]) - 1
                
                self.h0[:,:] = hdf5_in['h0'][nt,:,:]
                self.h1[:,:] = hdf5_in['h1'][nt,:,:]
                self.h2[:,:] = hdf5_in['h2'][nt,:,:]
                
                self.h[:,:] = self.h0 + self.h1 + self.h2
                
                for ih in range(0, self.nhist):
                    self.history0[:,:,ih] = hdf5_in['h0'][nt-ih,:,:]
                    self.history1[:,:,ih] = hdf5_in['h1'][nt-ih,:,:]
                    self.history2[:,:,ih] = hdf5_in['h2'][nt-ih,:,:]
                    
                    self.history[:,:,ih] = self.history0[:,:,ih] + self.history1[:,:,ih] + self.history2[:,:,ih]
                
                self.Ekin  = (hdf5_in['f'][nt,:,:] * self.h0).sum() * self.grid.hx * self.grid.hv
                self.Epot  = (hdf5_in['f'][nt,:,:] * self.h1).sum() * self.grid.hx * self.grid.hv \
                           + (hdf5_in['f'][nt,:,:] * self.h2).sum() * self.grid.hx * self.grid.hv
                
                self.Ekin0 = (hdf5_in['f'][ 0,:,:] * hdf5_in['h0'][0,:,:]).sum() * self.grid.hx * self.grid.hv
                self.Epot0 = (hdf5_in['f'][ 0,:,:] * hdf5_in['h1'][0,:,:]).sum() * self.grid.hx * self.grid.hv \
                           + (hdf5_in['f'][ 0,:,:] * hdf5_in['h2'][0,:,:]).sum() * self.grid.hx * self.grid.hv
                
                self.E  = self.Ekin  + 0.5 * self.Epot
                self.E0 = self.Ekin0 + 0.5 * self.Epot0
                
                self.P0 = self.mass * (hdf5_in['f'][0,:,:].sum(axis=0) * self.grid.vGrid).sum() * self.grid.hx * self.grid.hv
                
                
                self.calculate_energy_error()
                self.calculate_momentum_error()
        
                if self.hdf5_h0 != None and self.hdf5_h1 != None:
                    self.hdf5_h0[0,:,:] = hdf5_in['h0'][0,:,:]
                    self.hdf5_h1[0,:,:] = hdf5_in['h1'][0,:,:]
                    self.hdf5_h2[0,:,:] = hdf5_in['h2'][0,:,:]
                
            else:
                for iv in range(0, self.grid.nv):
                    self.h0[:,iv] = 0.5 * self.mass * self.grid.vGrid[iv]**2
                
                self.set_initial_state(potential, distribution)
    
    
    def set_distribution(self, f):
        self.f = f
        
    
    def set_initial_state(self, potential=None, distribution=None):
        if distribution != None:
            self.set_distribution(distribution.f)
        
        self.update_potential(potential)
        self.copy_initial_data()
        
    
    def copy_initial_data(self):
        for ih in range(0, self.nhist):
            self.history [:,:,ih] = self.h
            self.history0[:,:,ih] = self.h0
            self.history1[:,:,ih] = self.h1
            self.history2[:,:,ih] = self.h2
        
        self.calculate_total_energy()
        self.calculate_total_momentum()
        
        self.Ekin0   = self.Ekin
        self.Epot0   = self.Epot
        self.E0      = self.E
        self.P0      = self.P
        self.E_error = 0.0
        self.P_error = 0.0
        
    
    def update(self, potential=None):
        self.update_potential(potential)
        self.update_history()
        
        self.calculate_total_energy()
        self.calculate_total_momentum()
        self.calculate_energy_error()
        self.calculate_momentum_error()
        
    
    def update_potential(self, potential):
        '''
        Resets the potential part of the Hamiltonian and
        updates the total Hamiltonian accordingly.
        '''
        if potential == None:
            self.h1[:,:] = 0.0
        else:
            for ix in range(0, self.grid.nx):
                self.h1[ix,:] = potential.phi[ix]
            
        self.h[:,:] = self.h0 + self.h1
        
    
    def update_history(self):
        new_ind = range(1, self.nhist)     # e.g. [1,2,3,4]
        old_ind = range(0, self.nhist-1)   # e.g. [0,1,2,3]
        
        self.history[:,:,new_ind] = self.history[:,:,old_ind]
        self.history[:,:,0      ] = self.h[:,:]
        
        self.history0[:,:,new_ind] = self.history0[:,:,old_ind]
        self.history0[:,:,0      ] = self.h0[:,:]
        
        self.history1[:,:,new_ind] = self.history1[:,:,old_ind]
        self.history1[:,:,0      ] = self.h1[:,:]
        
        self.history2[:,:,new_ind] = self.history2[:,:,old_ind]
        self.history2[:,:,0      ] = self.h2[:,:]
        
    
    def calculate_total_energy(self):
        '''
        Calculates total energy integral w.r.t. the given distribution function.
        '''
        
        cdef np.uint64_t nx = self.grid.nx
        cdef np.uint64_t nv = self.grid.nv
        
        cdef np.uint64_t ix, ixp, iv
        cdef np.float64_t Ekin, Epot
        
        h1_ave = self.h1.mean()
        h2_ave = self.h2.mean()
        
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.h0
        cdef np.ndarray[np.float64_t, ndim=2] h1 = self.h1 - h1_ave
        cdef np.ndarray[np.float64_t, ndim=2] h2 = self.h2 - h2_ave
        cdef np.ndarray[np.float64_t, ndim=2] f  = self.f
        
        Ekin = 0.0
        Epot = 0.0
        
        if self.f != None:
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
        
        self.Ekin = Ekin * self.grid.hx * self.grid.hv * 0.25 * 0.25
        self.Epot = Epot * self.grid.hx * self.grid.hv * 0.25 * 0.25
        
        
#        self.Ekin  = (f * h0).sum() * self.grid.hx * self.grid.hv
#        self.Epot  = (f * h1).sum() * self.grid.hx * self.grid.hv \
#                   + (f * h2).sum() * self.grid.hx * self.grid.hv
        
        
        self.E = self.Ekin + 0.5 * self.Epot
        
    
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
        
#        if self.f != None:
#            self.P = self.mass * (self.f.sum(axis=0) * self.grid.vGrid).sum() * self.grid.hx * self.grid.hv
#        else:
#            self.P = 0.0
        self.P = 0.0
        
    
    def calculate_momentum_error(self):
        '''
        Calculates momentum error, i.e. relative difference between P and P0.
        '''
        if self.P0 != 0.0:
            self.P_error = (self.P - self.P0) # / self.P0
        else:
            self.P_error = 0.0
    

    def set_hdf5_file(self, hdf5):
        self.hdf5_f  = hdf5['f']
        self.hdf5_h0 = hdf5['h0']
        self.hdf5_h1 = hdf5['h1']
        self.hdf5_h2 = hdf5['h2']
        
        
    def save_to_hdf5(self, iTime):
        if self.hdf5_h0 == None or self.hdf5_h1 == None or self.hdf5_h2 == None:
            return
        
        self.hdf5_h0[iTime,:,:] = self.h0
        self.hdf5_h1[iTime,:,:] = self.h1
        self.hdf5_h2[iTime,:,:] = self.h2
        
    
    def read_from_hdf5(self, iTime):
        self.h0[:,:] = self.hdf5_h0[iTime,:,:]
        self.h1[:,:] = self.hdf5_h1[iTime,:,:]
        self.h2[:,:] = self.hdf5_h2[iTime,:,:]
        self.h [:,:] = self.h0 + self.h1 + self.h2
        self.f  = self.hdf5_f[iTime,:,:]
        
        self.calculate_total_energy()
        self.calculate_total_momentum()
        self.calculate_energy_error()
        self.calculate_momentum_error()
        
    
