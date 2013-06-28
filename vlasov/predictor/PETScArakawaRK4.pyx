'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.Toolbox import Toolbox


cdef class PETScArakawaRK4(object):
    '''
    PETSc/Cython Implementation of Explicit Arakawa-RK4 Vlasov-Poisson Solver
    '''
    
    
    def __init__(self, VIDA da1, VIDA dax, Vec H0,
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
        self.X1 = self.da1.createGlobalVec()
        self.X2 = self.da1.createGlobalVec()
        self.X3 = self.da1.createGlobalVec()
        self.X4 = self.da1.createGlobalVec()
        
        # create local vectors
        self.localX  = da1.createLocalVec()
        self.localX1 = da1.createLocalVec()
        self.localX2 = da1.createLocalVec()
        self.localX3 = da1.createLocalVec()
        self.localX4 = da1.createLocalVec()
        self.localH0 = da1.createLocalVec()
        self.localH1 = da1.createLocalVec()
        
        # create toolbox object
        self.toolbox = Toolbox(da1, dax, v, nx, nv, ht, hx, hv)
        
     
    def rk4_J1(self, Vec X, Vec H1):
        
        cdef np.ndarray[np.float64_t, ndim=2] x
        cdef np.ndarray[np.float64_t, ndim=2] tx1
        cdef np.ndarray[np.float64_t, ndim=2] tx2
        cdef np.ndarray[np.float64_t, ndim=2] tx3
        cdef np.ndarray[np.float64_t, ndim=2] tx4
        
        cdef np.ndarray[np.float64_t, ndim=2] h0
        cdef np.ndarray[np.float64_t, ndim=2] h1
        
        h0  = self.da1.getLocalArray(self.H0, self.localH0)
        h1  = self.da1.getLocalArray(H1,      self.localH1)
        x   = self.da1.getLocalArray(X,       self.localX )
        
        tx1 = self.da1.getGlobalArray(self.X1)
        self.toolbox.arakawa_J1_timestep(x, tx1, h0, h1)
        
        tx1 = self.da1.getLocalArray(self.X1, self.localX1)
        tx2 = self.da1.getGlobalArray(self.X2)
        self.toolbox.arakawa_J1_timestep(x + 0.5 * self.ht * tx1, tx2, h0, h1)
        
        tx2 = self.da1.getLocalArray(self.X2, self.localX2)
        tx3 = self.da1.getGlobalArray(self.X3)
        self.toolbox.arakawa_J1_timestep(x + 0.5 * self.ht * tx2, tx3, h0, h1)
        
        tx3 = self.da1.getLocalArray(self.X3, self.localX3)
        tx4 = self.da1.getGlobalArray(self.X4)
        self.toolbox.arakawa_J1_timestep(x + 1.0 * self.ht * tx3, tx4, h0, h1)
        
        x   = self.da1.getGlobalArray(X)
        tx1 = self.da1.getGlobalArray(self.X1)
        tx2 = self.da1.getGlobalArray(self.X2)
        tx3 = self.da1.getGlobalArray(self.X3)
        tx4 = self.da1.getGlobalArray(self.X4)
        
        x[:,:] = x + self.ht * (tx1 + 2.*tx2 + 2.*tx3 + tx4) / 6.
        

    def rk4_J2(self, Vec X, Vec H1):
        
        cdef np.ndarray[np.float64_t, ndim=2] x
        cdef np.ndarray[np.float64_t, ndim=2] tx1
        cdef np.ndarray[np.float64_t, ndim=2] tx2
        cdef np.ndarray[np.float64_t, ndim=2] tx3
        cdef np.ndarray[np.float64_t, ndim=2] tx4
        
        cdef np.ndarray[np.float64_t, ndim=2] h0
        cdef np.ndarray[np.float64_t, ndim=2] h1
        
        h0  = self.da1.getLocalArray(self.H0, self.localH0)
        h1  = self.da1.getLocalArray(H1,      self.localH1)
        x   = self.da1.getLocalArray(X,       self.localX )
        
        tx1 = self.da1.getGlobalArray(self.X1)
        self.toolbox.arakawa_J2_timestep(x, tx1, h0, h1)
        
        tx1 = self.da1.getLocalArray(self.X1, self.localX1)
        tx2 = self.da1.getGlobalArray(self.X2)
        self.toolbox.arakawa_J2_timestep(x + 0.5 * self.ht * tx1, tx2, h0, h1)
        
        tx2 = self.da1.getLocalArray(self.X2, self.localX2)
        tx3 = self.da1.getGlobalArray(self.X3)
        self.toolbox.arakawa_J2_timestep(x + 0.5 * self.ht * tx2, tx3, h0, h1)
        
        tx3 = self.da1.getLocalArray(self.X3, self.localX3)
        tx4 = self.da1.getGlobalArray(self.X4)
        self.toolbox.arakawa_J2_timestep(x + 1.0 * self.ht * tx3, tx4, h0, h1)
        
        x   = self.da1.getGlobalArray(X)
        tx1 = self.da1.getGlobalArray(self.X1)
        tx2 = self.da1.getGlobalArray(self.X2)
        tx3 = self.da1.getGlobalArray(self.X3)
        tx4 = self.da1.getGlobalArray(self.X4)
        
        x[:,:] = x + self.ht * (tx1 + 2.*tx2 + 2.*tx3 + tx4) / 6.
        

    def rk4_J4(self, Vec X, Vec H1):
        
        cdef np.ndarray[np.float64_t, ndim=2] x
        cdef np.ndarray[np.float64_t, ndim=2] tx1
        cdef np.ndarray[np.float64_t, ndim=2] tx2
        cdef np.ndarray[np.float64_t, ndim=2] tx3
        cdef np.ndarray[np.float64_t, ndim=2] tx4
        
        cdef np.ndarray[np.float64_t, ndim=2] h0
        cdef np.ndarray[np.float64_t, ndim=2] h1
        
        h0  = self.da1.getLocalArray(self.H0, self.localH0)
        h1  = self.da1.getLocalArray(H1,      self.localH1)
        x   = self.da1.getLocalArray(X,       self.localX )
        
        tx1 = self.da1.getGlobalArray(self.X1)
        self.toolbox.arakawa_J4_timestep(x, tx1, h0, h1)
        
        tx1 = self.da1.getLocalArray(self.X1, self.localX1)
        tx2 = self.da1.getGlobalArray(self.X2)
        self.toolbox.arakawa_J4_timestep(x + 0.5 * self.ht * tx1, tx2, h0, h1)
        
        tx2 = self.da1.getLocalArray(self.X2, self.localX2)
        tx3 = self.da1.getGlobalArray(self.X3)
        self.toolbox.arakawa_J4_timestep(x + 0.5 * self.ht * tx2, tx3, h0, h1)
        
        tx3 = self.da1.getLocalArray(self.X3, self.localX3)
        tx4 = self.da1.getGlobalArray(self.X4)
        self.toolbox.arakawa_J4_timestep(x + 1.0 * self.ht * tx3, tx4, h0, h1)
        
        x   = self.da1.getGlobalArray(X)
        tx1 = self.da1.getGlobalArray(self.X1)
        tx2 = self.da1.getGlobalArray(self.X2)
        tx3 = self.da1.getGlobalArray(self.X3)
        tx4 = self.da1.getGlobalArray(self.X4)
        
        x[:,:] = x + self.ht * (tx1 + 2.*tx2 + 2.*tx3 + tx4) / 6.
        

