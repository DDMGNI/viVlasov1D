'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.solvers.components.PoissonBracket import PoissonBracket


cdef class PETScArakawaSymplectic(object):
    '''
    PETSc/Cython Implementation of Explicit Arakawa-Symplectic-Splitting Vlasov-Poisson Solver
    '''
    
    
    def __init__(self, 
                 config    not None,
                 object da1  not None,
                 Grid grid not None,
                 Vec H0    not None,
                 Vec H1    not None,
                 Vec H2    not None,
                 niter=1):
        '''
        Constructor
        '''
        
        # number of iterations per timestep
        self.niter = niter
        
        # distributed array and grid
        self.da1  = da1
        self.grid = grid
        
        # Hamiltonians
        self.H0 = H0
        self.H1 = H1
        self.H2 = H2
        
        # create global vectors
        self.Y = da1.createGlobalVec()
        
        # create local vectors
        self.localH0 = da1.createLocalVec()
        self.localH1 = da1.createLocalVec()
        self.localH2 = da1.createLocalVec()
        
        self.localX  = da1.createLocalVec()
        
        # create toolbox object
        self.arakawa = PoissonBracket.create(config.get_poisson_bracket(), da1, grid)
        
    
    
    def kinetic(self, Vec X, np.float64_t factor=0.5):
        
        cdef np.ndarray[np.float64_t, ndim=2] x
        cdef np.ndarray[np.float64_t, ndim=2] y
        cdef np.ndarray[np.float64_t, ndim=2] h0
        
        h0 = getLocalArray(self.da1, self.H0, self.localH0)
        x  = getLocalArray(self.da1, X,       self.localX)
        y  = getGlobalArray(self.da1, self.Y)
        
        self.arakawa.poisson_bracket_array(x, y, h0, 1.0)
        
        x  = getGlobalArray(self.da1, X)
        y  = getGlobalArray(self.da1, self.Y)
        
        x[:,:] += factor * self.grid.ht / float(self.niter) * y


    def potential(self, Vec X, np.float64_t factor=1.0):
        
        cdef np.ndarray[np.float64_t, ndim=2] x
        cdef np.ndarray[np.float64_t, ndim=2] y
        cdef np.ndarray[np.float64_t, ndim=2] h1
        cdef np.ndarray[np.float64_t, ndim=2] h2
        
        h1 = getLocalArray(self.da1, self.H1, self.localH1)
        h2 = getLocalArray(self.da1, self.H2, self.localH2)
        x  = getLocalArray(self.da1, X,       self.localX)
        y  = getGlobalArray(self.da1, self.Y)
        
        self.arakawa.poisson_bracket_array(x, y, h1+h2, 1.0)
        
        x  = getGlobalArray(self.da1, X)
        y  = getGlobalArray(self.da1, self.Y)
        
        x[:,:] += factor * self.grid.ht / float(self.niter) * y


