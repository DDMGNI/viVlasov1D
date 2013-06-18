'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport Mat, Vec


cdef class PETScPoissonSolver(object):
    '''
    
    '''
    
    def __init__(self, VIDA da1, VIDA dax, 
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t hx, np.float64_t hv,
                 np.float64_t charge):
        '''
        Constructor
        '''
        
        # disstributed array
        self.dax = dax
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.hx = hx
        self.hv = hv
        
        self.hx2     = hx**2
        self.hx2_inv = 1. / self.hx2 
        
        # poisson constant
        self.charge = charge
        
        # create local vectors
        self.localX = dax.createLocalVec()
        self.localN = dax.createLocalVec()
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
    
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef np.int64_t i, ix, iy
        cdef np.int64_t xe, xs
        
        cdef np.float64_t phisum
        
        (xs, xe), = self.dax.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=1] y = self.dax.getGlobalArray(Y)
        cdef np.ndarray[np.float64_t, ndim=1] x = self.dax.getLocalArray(X, self.localX)
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            y[iy] = (2. * x[ix] - x[ix-1] - x[ix+1]) * self.hx2_inv
        
    
    @cython.boundscheck(False)
    def formRHS(self, Vec N, Vec B):
        cdef np.int64_t i, ix, iy
        cdef np.int64_t xs, xe
        
        cdef np.float64_t nmean = N.sum() / self.nx
        
        cdef np.ndarray[np.float64_t, ndim=1] b = self.dax.getGlobalArray(B)
        cdef np.ndarray[np.float64_t, ndim=1] n = self.dax.getLocalArray(N, self.localN)
        
        
        (xs, xe), = self.dax.getRanges()
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            b[iy] = - ( n[ix] - nmean) * self.charge
        
