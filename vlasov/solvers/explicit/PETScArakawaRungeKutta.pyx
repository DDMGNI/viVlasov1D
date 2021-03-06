'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Vec


cdef class PETScArakawaRungeKutta(PETScExplicitSolver):
    '''
    PETSc/Cython Implementation of Explicit Arakawa-RK4 Vlasov-Poisson Solver
    '''
    
    def rk4_16(self, Vec X, Vec H1):
        
        cdef np.ndarray[np.float64_t, ndim=2] x
        cdef np.ndarray[np.float64_t, ndim=2] tx1
        cdef np.ndarray[np.float64_t, ndim=2] tx2
        cdef np.ndarray[np.float64_t, ndim=2] tx3
        cdef np.ndarray[np.float64_t, ndim=2] tx4
        
        cdef np.ndarray[np.float64_t, ndim=2] h0  = getLocalArray(self.da1, self.H0, self.localH0)
        cdef np.ndarray[np.float64_t, ndim=2] h1  = getLocalArray(self.da1, self.H1, self.localH1)
        cdef np.ndarray[np.float64_t, ndim=2] h2  = getLocalArray(self.da1, self.H2, self.localH2)

        cdef np.ndarray[np.float64_t, ndim=2] h   = h0 + h1 + h2
        
        
        x   = getLocalArray(self.da1, X,       self.localX )
        tx1 = getGlobalArray(self.da1, self.X1)
        self.arakawa.poisson_bracket_array(x, tx1, h, 1.0)
        
        tx1 = getLocalArray(self.da1, self.X1, self.localX1)
        tx2 = getGlobalArray(self.da1, self.X2)
        self.arakawa.poisson_bracket_array(x + 0.5 * self.grid.ht / float(self.niter) * tx1, tx2, h, 1.0)
        
        tx2 = getLocalArray(self.da1, self.X2, self.localX2)
        tx3 = getGlobalArray(self.da1, self.X3)
        self.arakawa.poisson_bracket_array(x + 0.5 * self.grid.ht / float(self.niter) * tx2, tx3, h, 1.0)
        
        tx3 = getLocalArray(self.da1, self.X3, self.localX3)
        tx4 = getGlobalArray(self.da1, self.X4)
        self.arakawa.poisson_bracket_array(x + 1.0 * self.grid.ht / float(self.niter) * tx3, tx4, h, 1.0)
        
        x   = getGlobalArray(self.da1, X)
        tx1 = getGlobalArray(self.da1, self.X1)
        tx2 = getGlobalArray(self.da1, self.X2)
        tx3 = getGlobalArray(self.da1, self.X3)
        tx4 = getGlobalArray(self.da1, self.X4)
        
        x[:,:] = x + self.grid.ht / float(self.niter) * (tx1 + 2.*tx2 + 2.*tx3 + tx4) / 6.
        


    def rk4_18(self, Vec X, Vec H1):
        
        cdef np.ndarray[np.float64_t, ndim=2] x
        cdef np.ndarray[np.float64_t, ndim=2] tx1
        cdef np.ndarray[np.float64_t, ndim=2] tx2
        cdef np.ndarray[np.float64_t, ndim=2] tx3
        cdef np.ndarray[np.float64_t, ndim=2] tx4
        
        cdef np.ndarray[np.float64_t, ndim=2] h0  = getLocalArray(self.da1, self.H0, self.localH0)
        cdef np.ndarray[np.float64_t, ndim=2] h1  = getLocalArray(self.da1, self.H1, self.localH1)
        cdef np.ndarray[np.float64_t, ndim=2] h2  = getLocalArray(self.da1, self.H2, self.localH2)

        cdef np.ndarray[np.float64_t, ndim=2] h   = h0 + h1 + h2
        
        
        x   = getLocalArray(self.da1, X,       self.localX )
        tx1 = getGlobalArray(self.da1, self.X1)
        self.arakawa.poisson_bracket_array(x, tx1, h, 1.0)
        
        tx1 = getLocalArray(self.da1, self.X1, self.localX1)
        tx2 = getGlobalArray(self.da1, self.X2)
        self.arakawa.poisson_bracket_array(x + 1./3. * self.grid.ht / float(self.niter) * tx1, tx2, h, 1.0)
        
        tx2 = getLocalArray(self.da1, self.X2, self.localX2)
        tx3 = getGlobalArray(self.da1, self.X3)
        self.arakawa.poisson_bracket_array(x - 1./3. * self.grid.ht / float(self.niter) * tx1 + self.grid.ht / float(self.niter) * tx2, tx3, h, 1.0)
        
        tx3 = getLocalArray(self.da1, self.X3, self.localX3)
        tx4 = getGlobalArray(self.da1, self.X4)
        self.arakawa.poisson_bracket_array(x + self.grid.ht / float(self.niter) * tx1 - self.grid.ht / float(self.niter) * tx2 + self.grid.ht / float(self.niter) * tx3, tx4, h, 1.0)
        
        x   = getGlobalArray(self.da1, X)
        tx1 = getGlobalArray(self.da1, self.X1)
        tx2 = getGlobalArray(self.da1, self.X2)
        tx3 = getGlobalArray(self.da1, self.X3)
        tx4 = getGlobalArray(self.da1, self.X4)
        
        x[:,:] = x + self.grid.ht / float(self.niter) * (tx1 + 3.*tx2 + 3.*tx3 + tx4) / 8.
        

