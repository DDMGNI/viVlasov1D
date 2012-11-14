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
        
        assert da1.getDim() == 2
        assert dax.getDim() == 1
        
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
        
        # create local vectors
        self.localB = dax.createLocalVec()
        self.localX = dax.createLocalVec()
        self.localF = da1.createLocalVec()
        
    
    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, ix, iy
        cdef np.uint64_t xe, xs
        
        cdef np.float64_t phisum
        
        (xs, xe), = self.dax.getRanges()
        
        self.dax.globalToLocal(X, self.localX)
        
        cdef np.ndarray[np.float64_t, ndim=1] y = self.dax.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=1] x = self.dax.getVecArray(self.localX)[...]
        
        
        phisum = X.sum()
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
#            if i == 0:
#                y[iy] = phisum
#            else:            
#                y[iy] = (2. * x[ix] - x[ix-1] - x[ix+1]) / self.hx**2
            y[iy] = (2. * x[ix] - x[ix-1] - x[ix+1])
        
    
    @cython.boundscheck(False)
    def formRHS(self, Vec F, Vec B):
        cdef np.uint64_t i, ix, iy
        cdef np.uint64_t xs, xe
        
        cdef np.float64_t fsum
        
        self.da1.globalToLocal(F, self.localF)
        
        cdef np.ndarray[np.float64_t, ndim=1] b = self.dax.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] f = self.da1.getVecArray(self.localF)[...]
        
        fsum = F.sum() * self.hv / self.nx
        
        (xs, xe), = self.dax.getRanges()
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
#            if i == 0:
#                # impose constraint
#                b[iy] = 0.
#            
#            else:
            integral = ( \
                         + 1. * f[ix-1, :].sum() \
                         + 2. * f[ix,   :].sum() \
                         + 1. * f[ix+1, :].sum() \
                       ) * 0.25 * self.hv
            
            b[iy] = - (integral - fsum) * self.poisson_const * self.hx**2
        
