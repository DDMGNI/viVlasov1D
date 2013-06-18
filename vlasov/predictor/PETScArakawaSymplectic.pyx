'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.Toolbox import Toolbox


cdef class PETScArakawaSymplectic(object):
    '''
    PETSc/Cython Implementation of Explicit Arakawa-RK4 Vlasov-Poisson Solver
    '''
    
    
    def __init__(self, VIDA da1, VIDA da2, VIDA dax, Vec H0,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv):
        '''
        Constructor
        '''
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv

        # distributed array
        self.da1 = da1
        
        # velocity grid
        self.v = v.copy()
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # create global vectors
        self.Y = da1.createGlobalVec()
        
        # create local vectors
        self.localX = da1.createLocalVec()
        self.localH = da1.createLocalVec()
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
    
    
    def kinetic(self, Vec X, np.float64_t factor=0.5):
        
        cdef np.ndarray[np.float64_t, ndim=2] x
        cdef np.ndarray[np.float64_t, ndim=2] y
        cdef np.ndarray[np.float64_t, ndim=2] h0
        
        h0 = self.da1.getLocalArray(self.H0, self.localH)
        x  = self.da1.getLocalArray(X,       self.localX)
        y  = self.da1.getGlobalArray(self.Y)
        
        self.toolbox.arakawa_J4_timestep_h(x, y, h0)
        
        x  = self.da1.getGlobalArray(X)
        y  = self.da1.getGlobalArray(self.Y)
        
        x[:,:] += factor * self.ht * y


    def potential(self, Vec X, Vec H1, np.float64_t factor=1.0):
        
        cdef np.ndarray[np.float64_t, ndim=2] x
        cdef np.ndarray[np.float64_t, ndim=2] y
        cdef np.ndarray[np.float64_t, ndim=2] h1
        
        h1 = self.da1.getLocalArray(H1, self.localH)
        x  = self.da1.getLocalArray(X,  self.localX)
        
        y  = self.da1.getGlobalArray(self.Y)
        self.toolbox.arakawa_J4_timestep_h(x, y, h1)
        
        x  = self.da1.getGlobalArray(X)
        y  = self.da1.getGlobalArray(self.Y)
        
        x[:,:] += factor * self.ht * y


