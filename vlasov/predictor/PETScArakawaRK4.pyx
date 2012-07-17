'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport DA, Vec

from vlasov.predictor.PETScArakawa import PETScArakawa


cdef class PETScArakawaRK4(object):
    '''
    PETSc/Cython Implementation of Explicit Arakawa-RK4 Vlasov-Poisson Solver
    '''
    
    
    def __cinit__(self, DA da1, DA da2, Vec H0,
                  np.uint64_t nx, np.uint64_t nv,
                  np.float64_t ht, np.float64_t hx, np.float64_t hv,
                  PETScArakawa arakawa = None):
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
        self.da2 = da2
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # create global vectors
        self.X1 = self.da1.createGlobalVec()
        self.X2 = self.da1.createGlobalVec()
        self.X3 = self.da1.createGlobalVec()
        self.X4 = self.da1.createGlobalVec()
        
        # create local vectors
        self.localX  = da2.createLocalVec()
        self.localX1 = da1.createLocalVec()
        self.localX2 = da1.createLocalVec()
        self.localX3 = da1.createLocalVec()
        self.localX4 = da1.createLocalVec()
        self.localH0 = da1.createLocalVec()
        
        # create Arakawa solver object
        if arakawa != None:
            self.arakawa = arakawa
        else:
            self.arakawa = PETScArakawa(da1, da2, nx, nv, hx, hv)
        
     
    def rk4(self, Vec X):
        
        cdef np.ndarray[np.float64_t, ndim=2] tx
        cdef np.ndarray[np.float64_t, ndim=2] tx1
        cdef np.ndarray[np.float64_t, ndim=2] tx2
        cdef np.ndarray[np.float64_t, ndim=2] tx3
        cdef np.ndarray[np.float64_t, ndim=2] tx4
        
        cdef np.ndarray[np.float64_t, ndim=2] h0
        cdef np.ndarray[np.float64_t, ndim=2] h1
        
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da2.globalToLocal(X,       self.localX)
        
        h0  = self.da1.getVecArray(self.localH0)[...]
        x   = self.da2.getVecArray(self.localX )[...]
        tx  = x[:,:,0]
        h1  = x[:,:,1]
        
        tx1 = self.da1.getVecArray(self.X1)[...]
        self.arakawa.arakawa_timestep(tx, tx1, h0, h1)
        
        self.da1.globalToLocal(self.X1, self.localX1)
        tx1 = self.da1.getVecArray(self.localX1)[...]
        tx2 = self.da1.getVecArray(self.X2)[...]
        self.arakawa.arakawa_timestep(tx + 0.5 * self.ht * tx1, tx2, h0, h1)
        
        self.da1.globalToLocal(self.X2, self.localX2)
        tx2 = self.da1.getVecArray(self.localX2)[...]
        tx3 = self.da1.getVecArray(self.X3)[...]
        self.arakawa.arakawa_timestep(tx + 0.5 * self.ht * tx2, tx3, h0, h1)
        
        self.da1.globalToLocal(self.X3, self.localX3)
        tx3 = self.da1.getVecArray(self.localX3)[...]
        tx4 = self.da1.getVecArray(self.X4)[...]
        self.arakawa.arakawa_timestep(tx + 1.0 * self.ht * tx3, tx4, h0, h1)
        
        tx  = self.da2.getVecArray(X)[...][:,:,0]
        tx1 = self.da1.getVecArray(self.X1)[...]
        tx2 = self.da1.getVecArray(self.X2)[...]
        tx3 = self.da1.getVecArray(self.X3)[...]
        tx4 = self.da1.getVecArray(self.X4)[...]
        
        tx[:,:] = tx + self.ht * (tx1 + 2.*tx2 + 2.*tx3 + tx4) / 6.
        
