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


cdef class PETScPoissonSolverBase(object):
    '''
    
    '''
    
    def __init__(self, VIDA dax, 
                 np.uint64_t nx, np.float64_t hx,
                 np.float64_t charge):
        '''
        Constructor
        '''
        
        # distributed array
        self.dax = dax
        
        # grid
        self.nx = nx
        self.hx = hx
        
        self.hx2     = hx**2
        self.hx2_inv = 1. / self.hx2 
        
        # poisson constant
        self.charge = charge
        
        # create local vectors
        self.localX = dax.createLocalVec()
        self.localN = dax.createLocalVec()
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        print("ERROR: PETScPoissonSolver function not implemented!")
    
    
    def formMat(self, Mat A):
        print("ERROR: PETScPoissonSolver function not implemented!")
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def formRHS(self, Vec N, Vec B):
        cdef int i, ix, iy
        cdef int xs, xe, sw
        
        cdef double nmean = N.sum() / self.nx
        
        cdef double[:] b = self.dax.getGlobalArray(B)
        cdef double[:] n = self.dax.getLocalArray(N, self.localN)
        
        (xs, xe), = self.dax.getRanges()
        sw        = self.dax.getStencilWidth()
        
        for i in range(xs, xe):
            ix = i-xs+sw
            iy = i-xs
            
            b[iy] = - (n[ix] - nmean) * self.charge
        

    def matrix_mult(self, Vec X, Vec Y):
        print("ERROR: PETScPoissonSolver function not implemented!")
        
    
    def function_mult(self, Vec X, Vec N, Vec Y):
        print("ERROR: PETScPoissonSolver function not implemented!")
