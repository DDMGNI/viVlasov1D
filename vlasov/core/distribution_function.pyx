'''
Created on Mar 20, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import  numpy as np
cimport numpy as np

from libc.math cimport abs, log, pow

from vlasov.toolbox.boltzmann import boltzmannian_grid
from vlasov.toolbox.maxwell   import maxwellian_grid


class DistributionFunction(object):
    '''
    Discrete representation of the distribution function of the Vlasov-Poisson system.
    '''


    def __init__(self, grid, mass=1.0, hdf5=None):
        '''
        Constructor
        
        Temperature and density may be scalar for constant profiles or ndarray[nx].
        '''
        
        self.grid  = grid
        self.mass  = mass
        
        self.hdf5_f = None
        self.hdf5_n = None
        
        self.fMin   = 0.0
        self.fMax   = 0.0
        
        self.N  = 0.0           # current total particle number (L1 norm)
        self.N0 = 0.0           # initial total particle number
        self.N_error = 0.0      # error in total particle number (N-N0)/N0
        
        self.L1   = 0.0         # current L1 norm
        self.L1_0 = 0.0         # initial L1 norm
        self.L1_error = 0.0     # error in L1 norm
        
        self.L2   = 0.0         # current L2 norm
        self.L2_0 = 0.0         # initial L2 norm
        self.L2_error = 0.0     # error in L2 norm
        
        self.L3   = 0.0         # current L3 norm
        self.L3_0 = 0.0         # initial L3 norm
        self.L3_error = 0.0     # error in L3 norm
        
        self.L4   = 0.0         # current L4 norm
        self.L4_0 = 0.0         # initial L4 norm
        self.L4_error = 0.0     # error in L4 norm
        
        self.L5   = 0.0         # current L5 norm
        self.L5_0 = 0.0         # initial L5 norm
        self.L5_error = 0.0     # error in L5 norm
        
        self.L6   = 0.0         # current L6 norm
        self.L6_0 = 0.0         # initial L6 norm
        self.L6_error = 0.0     # error in L6 norm
        
        self.L8   = 0.0         # current L8 norm
        self.L8_0 = 0.0         # initial L8 norm
        self.L8_error = 0.0     # error in L8 norm
        
        self.Lmin   = 0.0       # current Lmin norm
        self.Lmin_0 = 0.0       # initial Lmin norm
        self.Lmin_error = 0.0   # error in Lmin norm
        
        self.Lmax   = 0.0       # current Lmax norm
        self.Lmax_0 = 0.0       # initial Lmax norm
        self.Lmax_error = 0.0   # error in Lmax norm
        
        self.S  = 0.0           # current total entropy
        self.S0 = 0.0           # initial total entropy
        self.S_error = 0.0      # error in total entropy (S-S0)/S0
        
        
        self.temperature = np.zeros(self.grid.nx)
        self.density     = np.zeros(self.grid.nx)
        
        
        self.f = None
        
        self.set_hdf5_file(hdf5)
        
        self.f0   = hdf5['f'][0,:,:].T
        self.fMin = hdf5['f'][:,:,:].min()
        self.fMax = hdf5['f'][:,:,:].max()
        
        self.fmin  = self.hdf5_f[0,:,:].min()
        self.fmax  = self.hdf5_f[0,:,:].max()
        self.fmax0 = self.fmax
        self.fmax_error = 0.
        
        self.update_integrals(self.f0)
        self.read_from_hdf5(0)
            
    
    
    def update_integrals(self, f0=None):
        self.calculate_density(f0)
        
        self.calculate_total_particle_number(f0)
        self.calculate_total_entropy(f0)
        self.calculate_norm(f0)
        
        if f0 == None:
            self.calculate_particle_number_error()
            self.calculate_entropy_error()
            self.calculate_norm_error()
        
    
    def calculate_density(self, f0=None):
        if f0 == None:
            f = self.f
        else:
            f = f0
        
        self.density[:] = f.sum(axis=1) * self.grid.hv
        
            
    
    def calculate_total_particle_number(self, f0=None):
        cdef np.uint64_t nx = self.grid.nx
        cdef np.uint64_t nv = self.grid.nv
        
        cdef np.uint64_t ix, ixp, iv
        cdef np.float64_t N
        
        cdef np.ndarray[np.float64_t, ndim=2] f
        
        
        N = 0.0
        
        if f0 == None:
            f = self.f
        else:
            f = f0
            
        for ix in range(0, nx):
            ixp = (ix+1) % nx
            
            for iv in range(0, nv-1):
                N += (f[ix,iv] + f[ixp,iv] + f[ixp,iv+1] + f[ix,iv+1])
            
        
        if f0 == None:
            self.N  = N * 0.25 * self.grid.hx * self.grid.hv
        else:
            self.N0 = N * 0.25 * self.grid.hx * self.grid.hv
        
    
    def calculate_particle_number_error(self):
        if self.N0 != 0.0:
            self.N_error = (self.N - self.N0) / self.N0
        else:
            self.N_error = 0.0
        
    
    def calculate_norm(self, np.ndarray[np.float64_t, ndim=2] f0=None):
        cdef np.uint64_t nx = self.grid.nx
        cdef np.uint64_t nv = self.grid.nv
        
        cdef np.uint64_t ix, ixp, iv
        cdef np.float64_t L1, L2, L3, L4, L5, L6, L8
        
        cdef np.ndarray[np.float64_t, ndim=2] f = self.f
        
        L1 = 0.0
        L2 = 0.0
        L3 = 0.0
        L4 = 0.0
        L5 = 0.0
        L6 = 0.0
        L8 = 0.0
        
        
        if f0 == None:
            for ix in range(0, nx):
                ixp = (ix+1) % nx
                
                for iv in range(0, nv-1):
                    L1 += abs(f[ix,iv])
                    L2 += pow(f[ix,iv], 2)
                    L3 += pow(f[ix,iv], 3)
                    L4 += pow(f[ix,iv], 4)
                    L5 += pow(f[ix,iv], 5)
                    L6 += pow(f[ix,iv], 6)
                    L8 += pow(f[ix,iv], 8)
            
            self.L1 = (self.grid.hx * self.grid.hv) * L1
            self.L2 = (self.grid.hx * self.grid.hv) * L2
            self.L3 = (self.grid.hx * self.grid.hv) * L3
            self.L4 = (self.grid.hx * self.grid.hv) * L4
            self.L5 = (self.grid.hx * self.grid.hv) * L5
            self.L6 = (self.grid.hx * self.grid.hv) * L6
            self.L8 = (self.grid.hx * self.grid.hv) * L8
            
            self.Lmin = f.min() * self.grid.hx * self.grid.hv
            self.Lmax = f.max() * self.grid.hx * self.grid.hv
            
        else:
            for ix in range(0, nx):
                ixp = (ix+1) % nx
                
                for iv in range(0, self.grid.nv-1):
                    L1 += abs(f0[ix,iv])
                    L2 += pow(f0[ix,iv], 2)
                    L3 += pow(f0[ix,iv], 3)
                    L4 += pow(f0[ix,iv], 4)
                    L5 += pow(f0[ix,iv], 5)
                    L6 += pow(f0[ix,iv], 6)
                    L8 += pow(f0[ix,iv], 8)
            
            self.L1_0 = (self.grid.hx * self.grid.hv) * L1
            self.L2_0 = (self.grid.hx * self.grid.hv) * L2
            self.L3_0 = (self.grid.hx * self.grid.hv) * L3
            self.L4_0 = (self.grid.hx * self.grid.hv) * L4
            self.L5_0 = (self.grid.hx * self.grid.hv) * L5
            self.L6_0 = (self.grid.hx * self.grid.hv) * L6
            self.L8_0 = (self.grid.hx * self.grid.hv) * L8
            
            self.Lmin_0 = f0.min() * self.grid.hx * self.grid.hv
            self.Lmax_0 = f0.max() * self.grid.hx * self.grid.hv
        
    
    def calculate_norm_error(self):
        if self.L1_0 != 0.0:
            self.L1_error = (self.L1 - self.L1_0) / self.L1_0
        else:
            self.L1_error = 0.0
        
        if self.L2_0 != 0.0:
            self.L2_error = (self.L2 - self.L2_0) / self.L2_0
        else:
            self.L2_error = 0.0
        
        if self.L3_0 != 0.0:
            self.L3_error = (self.L3 - self.L3_0) / self.L3_0
        else:
            self.L3_error = 0.0
        
        if self.L4_0 != 0.0:
            self.L4_error = (self.L4 - self.L4_0) / self.L4_0
        else:
            self.L4_error = 0.0
        
        if self.L5_0 != 0.0:
            self.L5_error = (self.L5 - self.L5_0) / self.L5_0
        else:
            self.L5_error = 0.0
        
        if self.L6_0 != 0.0:
            self.L6_error = (self.L6 - self.L6_0) / self.L6_0
        else:
            self.L6_error = 0.0
        
        if self.L8_0 != 0.0:
            self.L8_error = (self.L8 - self.L8_0) / self.L8_0
        else:
            self.L8_error = 0.0
        
        if self.Lmin_0 != 0.0:
            self.Lmin_error = (self.Lmin - self.Lmin_0) / self.Lmin_0
        else:
            self.Lmin_error = 0.0
        
        if self.Lmax_0 != 0.0:
            self.Lmax_error = (self.Lmax - self.Lmax_0) / self.Lmax_0
        else:
            self.Lmax_error = 0.0
        
    
    def calculate_total_entropy(self, np.ndarray[np.float64_t, ndim=2] f0=None):
        cdef np.uint64_t nx = self.grid.nx
        cdef np.uint64_t nv = self.grid.nv
        
        cdef np.uint64_t ix, ixp, iv
        cdef np.float64_t S
        
        cdef np.ndarray[np.float64_t, ndim=2] f
        
        if f0 == None:
            f = self.f.copy()
        else:
            f = f0.copy()
        
        f[f <= 0.] = min((1E-25, self.f0[self.f0 > 0.].min()))


        S = 0.0
        
        for ix in range(0, nx):
            ixp = (ix+1) % nx
            
            for iv in range(0, nv-1):
                S += f[ix,  iv  ] * log( f[ix,iv] )
                
        self.S = - self.grid.hx * self.grid.hv * S
        
        if f0 != None:
            self.S0 = self.S
        
    
    def calculate_entropy_error(self):
        if self.S0 != 0.0:
            self.S_error = (self.S - self.S0) / self.S0
        else:
            self.S_error = 0.0
        
    
    def set_hdf5_file(self, hdf5):
        self.hdf5_f = hdf5['f']
        
        
    def read_from_hdf5(self, iTime):
        self.f = self.hdf5_f[iTime,:,:].T
        
        self.fmin = self.hdf5_f[iTime,:,:].min()
        self.fmax = self.hdf5_f[iTime,:,:].max()
        
        if self.fmax0 > 0:
            self.fmax_error = (self.fmax - self.fmax0) / self.fmax0
        else:
            self.fmax_error = 0.
        
        self.update_integrals()
        
    
