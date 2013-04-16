'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport DA, Mat, Vec


cdef class PETScPoissonSolver(object):
    '''
    
    '''
    
    def __init__(self, DA da1, DA dax, 
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t hx, np.float64_t hv,
                 np.float64_t poisson_const):
        '''
        Constructor
        '''
        
        # disstributed array
        self.da1 = da1
        self.dax = dax
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.hx = hx
        self.hv = hv
        
        # poisson constant
        self.poisson_const = poisson_const
        self.eps = 1.E-3
        
        # create local vectors
        self.localX = dax.createLocalVec()
        self.localF = da1.createLocalVec()
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
    
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef np.int64_t i, ix, iy
        cdef np.int64_t xe, xs
        
        cdef np.float64_t phisum
        
        (xs, xe), = self.dax.getRanges()
        
        self.dax.globalToLocal(X, self.localX)
        
        cdef np.ndarray[np.float64_t, ndim=1] y = self.dax.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=1] x = self.dax.getVecArray(self.localX)[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            y[iy] = (2. * x[ix] - x[ix-1] - x[ix+1]) / self.hx**2
        
    
    @cython.boundscheck(False)
    def formRHS(self, Vec F, Vec B):
        cdef np.int64_t i, ix, iy
        cdef np.int64_t xs, xe
        
        cdef np.float64_t fsum = F.sum() * self.hv / self.nx
        
        self.da1.globalToLocal(F, self.localF)
        
        cdef np.ndarray[np.float64_t, ndim=1] b = self.dax.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] f = self.da1.getVecArray(self.localF)[...]
        
        
        (xs, xe), = self.dax.getRanges()
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            integral = ( \
                         + 1. * f[ix-1, :].sum() \
                         + 2. * f[ix,   :].sum() \
                         + 1. * f[ix+1, :].sum() \
                       ) * 0.25 * self.hv
            
            b[iy] = - (integral - fsum) * self.poisson_const
        
