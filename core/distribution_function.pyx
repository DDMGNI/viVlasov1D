'''
Created on Mar 20, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import  numpy as np
cimport numpy as np

from libc.math cimport abs, log, pow

from data import boltzmannian_grid, maxwellian_grid


class DistributionFunction(object):
    '''
    Discrete representation of the distribution function of the Vlasov-Poisson system.
    '''


    def __init__(self, grid, mass=1.0, nhist=1, 
                 distribution=None, distribution_file=None, distribution_module=None,
                 density=None,      density_file=None,      density_module=None, 
                 temperature=None,  temperature_file=None,  temperature_module=None,
                 hamiltonian=None,  hdf5_in=None,  hdf5_out=None, replay=False):
        '''
        Constructor
        
        Temperature and density may be scalar for constant profiles or ndarray[nx].
        '''
        
        self.average_diagnostics = True
        
        
        self.grid  = grid
        self.nhist = nhist
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
        
        
        if hdf5_in != None and replay:
            self.f = None
            
            self.set_hdf5_file(hdf5_in)
            
            self.f0   = hdf5_in['f'][0,:,:]
            self.fMin = hdf5_in['f'][:,:,:].min()
            self.fMax = hdf5_in['f'][:,:,:].max()
            
            self.update_integrals(self.f0)
            self.read_from_hdf5(0)
            
        else:
            if hdf5_out != None:
                # create HDF5 fields
                self.hdf5_f = hdf5_out.create_dataset('f', (self.grid.nt+1, self.grid.nx, self.grid.nv), '=f8')
                self.hdf5_n = hdf5_out.create_dataset('n', (self.grid.nt+1, self.grid.nx), '=f8')
            
            # create data arrays
            self.f       = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )
            self.f0      = np.zeros( (self.grid.nx, self.grid.nv), dtype=np.float64 )
            self.history = np.zeros( (self.grid.nx, self.grid.nv, self.nhist), dtype=np.float64 )
            
            
            if hdf5_in != None:
#                nt = len(hdf5_in['grid']['t'][:]) - 1
                nt = len(hdf5_in['t'][:]) - 1
                
                self.f[:,:]  = hdf5_in['f'][nt,:,:]
                
                for ih in range(0, self.nhist):
                    self.history[:,:,ih] = hdf5_in['f'][nt-ih,:,:]
                
                f0 = hdf5_in['f'][0,:,:]
                n0 = hdf5_in['n'][0,:]
                
                self.f0[:,:] = f0
                
                self.update_integrals(f0)
                self.update_integrals()

                if self.hdf5_f != None and self.hdf5_n != None:
                    self.hdf5_f[0,:,:] = f0
                    self.hdf5_n[0,:]   = n0
                
            else:
                if distribution_module != None:
                    init_data = __import__("runs." + distribution_module, globals(), locals(), ['distribution'], 0)
                    self.f[:] = init_data.distribution(grid)
                elif distribution_file != None:
                    self.f[:] = np.loadtxt(distribution_file)
                elif distribution != None and distribution != 0.0:
                    self.f[:] = distribution
                else:
                    if density_module != None:
                        init_data = __import__("runs." + density_module, globals(), locals(), ['density'], 0)
                        tdensity = init_data.density(grid)
                    elif density_file != None:
                        tdensity = np.loadtxt(density_file)
                    elif density != None:
                        tdensity = np.empty(self.grid.nx)
                        tdensity[:] = density
                    else:
                        print("ERROR: Neither distribution function nor density specified.")
                        exit()
                    
                    if temperature_module != None:
                        init_data = __import__("runs." + temperature_module, globals(), locals(), ['temperature'], 0)
                        self.temperature[:] = init_data.temperature(grid)
                    elif temperature_file != None:
                        self.temperature[:] = np.loadtxt(temperature_file)
                    elif temperature != None:
                        self.temperature[:] = temperature
                    else:
                        print("ERROR: No temperature specified.")
                        exit()
                
                    self.set_initial_state(tdensity, hamiltonian)
            
                self.f0[:] = self.f.copy()
                self.copy_initial_data()
                self.update_integrals()
        
    
    def zero_out_small_values(self, threshold=1.0E-6):
        self.f[self.f < threshold] = 0.0
        
        self.copy_initial_data()
        self.update_integrals()
        
    
    def set_initial_state(self, density, hamiltonian=None):
        if hamiltonian == None:
            self.initialise_maxwellian()
        else:
            self.initialise_boltzmann(hamiltonian.h)
        
        if self.grid.is_dirichlet():
            self.f[:, 0] = 0.
            self.f[:,-1] = 0.
        
        self.normalise()
        
        for ix in range(0, self.grid.nx):
            self.f[ix,:] *= density[ix]
        
    
    def normalise(self):
        for ix in range(0, self.grid.nx):
            self.f[ix,:] /= self.f[ix].sum() * self.grid.hv 
        
    
    def initialise_maxwellian(self):
        assert self.temperature.min() > 0.0
        self.f[:,:] = maxwellian_grid(self.grid, self.temperature)
        
    
    def initialise_boltzmann(self, h):
        assert self.temperature.min() > 0.0
        self.f[:,:] = boltzmannian_grid(self.grid, self.temperature, h)
        
    
    def copy_initial_data(self):
        for ih in range(0, self.nhist):
            self.history[:,:,ih] = self.f[:,:]
        
        self.update_integrals(self.f0)
        
    
    def update(self):
        self.update_history()
        self.update_integrals()
        
    
    def update_distribution(self, distribution):
        assert distribution != None
        
        if distribution.ndim == 1:
            assert self.f.shape[0] * self.f.shape[1] == distribution.shape[0]
            self.f[:,:] = distribution.reshape((self.grid.nx, self.grid.nv))
        elif distribution.ndim == 2:
            assert self.f.shape == distribution.shape
            self.f[:,:] = distribution
        else:
            pass 
        
    
    def update_history(self):
        new_ind = range(1, self.nhist)     # e.g. [1,2,3,4]
        old_ind = range(0, self.nhist-1)   # e.g. [0,1,2,3]
        
        self.history[:,:,new_ind] = self.history[:,:,old_ind]
        self.history[:,:,0      ] = self.f[:,:]
        
    
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
        
#        self.density[:] = f.sum(axis=1) * self.grid.hv
        
        for i in range(0, self.grid.nx):
            im = (i - 1 + self.grid.nx) % self.grid.nx
            ip = (i + 1) % self.grid.nx
            
            self.density[i] = 0.25 * ( f[im].sum() + 2. * f[i].sum() + f[ip].sum() ) * self.grid.hv
        
            
    
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
            
        for ix in np.arange(0, nx):
#            ixp = (ix+1) % nx
            
            for iv in np.arange(0, nv-1):
                N += f[ix,iv]
#                N += (f[ix,iv] + f[ixp,iv] + f[ixp,iv+1] + f[ix,iv+1])
            
        
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
        cdef np.float64_t L1
        cdef np.float64_t L2
        
        cdef np.ndarray[np.float64_t, ndim=2] f = self.f
        
        L1 = 0.0
        L2 = 0.0
        
        if f0 == None:
            for ix in np.arange(0, nx):
                ixp = (ix+1) % nx
                
                for iv in np.arange(0, nv-1):
                    L1 += abs(f[ix,iv] + f[ixp,iv] + f[ixp,iv+1] + f[ix,iv+1])
                    L2 += pow(f[ix,iv] + f[ixp,iv] + f[ixp,iv+1] + f[ix,iv+1], 2)
            
            self.L1 = 0.25 * (self.grid.hx * self.grid.hv) * L1
            self.L2 = 0.25**2 * 0.5 * (self.grid.hx * self.grid.hv) * L2
            
            self.Lmin = f.min() * self.grid.hx * self.grid.hv
            self.Lmax = f.max() * self.grid.hx * self.grid.hv
            
        else:
            for ix in range(0, nx):
                ixp = (ix+1) % nx
                
                for iv in range(0, self.grid.nv-1):
                    L1 += abs(f0[ix,iv] + f0[ixp,iv] + f0[ixp,iv+1] + f0[ix,iv+1])
                    L2 += pow(f0[ix,iv] + f0[ixp,iv] + f0[ixp,iv+1] + f0[ix,iv+1], 2)
            
            self.L1_0 = 0.25 * (self.grid.hx * self.grid.hv) * L1
            self.L2_0 = 0.25**2 * 0.5 * (self.grid.hx * self.grid.hv) * L2
            
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
        
        for ix in np.arange(0, nx):
            ixp = (ix+1) % nx
            
            for iv in np.arange(0, nv-1):
                S += 0.25 * ( f[ix,  iv  ]
                            + f[ixp, iv  ]
                            + f[ixp, iv+1]
                            + f[ix,  iv+1]
                 ) * ( log( ( f[ix,  iv  ]
                            + f[ixp, iv  ]
                            + f[ixp, iv+1]
                            + f[ix,  iv+1]
                            ) * 0.25 ) )
        
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
        
        
    def save_to_hdf5(self, iTime):
        if self.hdf5_f == None or self.hdf5_n == None:
            return
        
        self.hdf5_f[iTime,:,:] = self.f
        self.hdf5_n[iTime,:]   = self.density
        
    
    def read_from_hdf5(self, iTime):
        self.f = self.hdf5_f[iTime,:,:]
        
        self.update_integrals()
        
    
