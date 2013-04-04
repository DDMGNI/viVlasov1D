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


    def __init__(self, grid, nhist=1, potential=None, potential_file=None, potential_module=None,
                 poisson_const=1.0, hdf5_in=None,  hdf5_out=None, replay=False):
        '''
        Constructor
        '''
        
        self.average_diagnostics = False
        
        
        self.grid  = grid
        self.nhist = nhist
        self.f     = None
        
        self.hdf5_phi = None 
        
        self.E0 = 0.0
        self.E  = 0.0
        self.E_error = 0.0
        self.Efield = 0.0
        
#        self.poisson = Poisson(grid, poisson_const)
        self.poisson_const = poisson_const
        
        if hdf5_in != None and replay:
            self.phi = None
            self.set_hdf5_file(hdf5_in)
            self.read_from_hdf5(0)
            self.E0 = self.E
            
        else:
            if hdf5_out != None:
                # create HDF5 fields
                self.hdf5_phi = hdf5_out.create_dataset('phi', (self.grid.nt+1, self.grid.nx), '=f8')
            
            self.phi     = np.empty( (self.grid.nx),             dtype=np.float64 )
            self.history = np.zeros( (self.grid.nx, self.nhist), dtype=np.float64 )
            
            
            if hdf5_in != None:
                nt = len(hdf5_in['grid']['t'][:]) - 1
                
                self.phi[:]  = hdf5_in['phi'][0,:]
                
                self.calculate_energy()
                self.calculate_momentum()
                
                self.E0 = self.E
                
                
                self.phi[:]  = hdf5_in['phi'][nt,:]
                
                for ih in range(0, self.nhist):
                    self.history[:,ih] = hdf5_in['phi'][nt-ih,:]
                
                self.calculate_energy()
                
                
                if self.hdf5_phi != None:
                    self.hdf5_phi[0,:] = hdf5_in['phi'][0,:]
                
            else:
                if potential_module != None:
                    init_data   = __import__("runs." + potential_module, globals(), locals(), ['potential'], 0)
                    self.phi[:] = init_data.potential(grid)
                elif potential_file != None:
                    self.phi[:] = np.loadtxt(potential_file)
                elif potential != None:
                    self.phi[:] = potential
                
                self.copy_initial_data()
        
    
    def set_distribution(self, distribution=None):
        self.f = distribution
        
    
    def solve(self, f):
        self.phi[:] = self.poisson.solve(f)
        self.update()
        
    
    def copy_initial_data(self):
        for ih in range(0, self.nhist):
            self.history[:,ih] = self.phi[:]
        
        self.calculate_energy()
        self.E0 = self.E
        
    
    def update(self, potential=None):
        if potential != None:
            self.update_potential(potential)
        
        self.update_history()
        self.calculate_energy()
    
    
    def update_potential(self, potential):
        assert potential != None
        assert potential.shape == self.phi.shape
        
        self.phi[:] = potential
        
    
    def update_history(self):
        new_ind = range(1, self.nhist)     # e.g. [1,2,3,4]
        old_ind = range(0, self.nhist-1)   # e.g. [0,1,2,3]
        
        self.history[:,new_ind] = self.history[:,old_ind]
        self.history[:,0      ] = self.phi[:]
        
     
    def calculate_energy(self):
        cdef np.uint64_t nx = self.grid.nx
        
        cdef np.uint64_t ix, ixp
        cdef np.float64_t E
        
        cdef np.ndarray[np.float64_t, ndim=1] phi  = self.phi
        cdef np.ndarray[np.float64_t, ndim=1] tphi = phi - phi.mean()
        
        E = 0.0
        
        for ix in np.arange(0, nx):
            ixp = (ix+1) % nx
            
            E += pow(tphi[ixp] - tphi[ix], 2)
        
        self.E = 0.5 * E / self.grid.hx               # / hx**2 for the square of the derivative
                                                      # * hx**1 for the integration
        
    
    def set_hdf5_file(self, hdf5):
        self.hdf5_phi = hdf5['phi']
                
        
    def save_to_hdf5(self, iTime):
        if self.hdf5_phi == None:
            return
        
        self.hdf5_phi[iTime,:] = self.phi
        
    
    def read_from_hdf5(self, iTime):
        self.phi = self.hdf5_phi[iTime,:]
        
        self.calculate_energy()
        
    
