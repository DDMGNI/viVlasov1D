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


cdef class PETScPoissonSolver(PETScPoissonSolverBase):
    '''
    
    '''
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def formMat(self, Mat A):
        cdef int i, j
        cdef int xs, xe
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        (xs, xe), = self.dax.getRanges()
        
        
        # Laplace operator
        for i in range(xs, xe):
            row.index = (i,)
            row.field = 0
            col.field = 0
            
            for index, value in [
                    ((i-2,), +  1. * self.hx2_inv / 12.),
                    ((i-1,), - 16. * self.hx2_inv / 12.),
                    ((i,  ), + 30. * self.hx2_inv / 12.),
                    ((i+1,), - 16. * self.hx2_inv / 12.),
                    ((i+2,), +  1. * self.hx2_inv / 12.),
                ]:
                
                col.index = index
                A.setValueStencil(row, col, value)
            
        
        A.assemble()
        
    
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
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef int i, ix, iy
        cdef int xs, xe, sw
        
        cdef double[:] y = self.dax.getGlobalArray(Y)
        cdef double[:] x = self.dax.getLocalArray(X, self.localX)
        
        (xs, xe), = self.dax.getRanges()
        sw        = self.dax.getStencilWidth()
        
        for i in range(xs, xe):
            ix = i-xs+sw
            iy = i-xs
            
            y[iy] = (1. * x[ix-2] - 16. * x[ix-1] + 30. * x[ix] - 16. * x[ix+1] + 1. * x[ix+2]) * self.hx2_inv / 12.
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function_mult(self, Vec X, Vec N, Vec Y):
        cdef int i, ix, iy
        cdef int xs, xe, sw
        
        cdef double nmean = N.sum() / self.nx
        
        cdef double[:] y = self.dax.getGlobalArray(Y)
        cdef double[:] x = self.dax.getLocalArray(X, self.localX)
        cdef double[:] n = self.dax.getLocalArray(N, self.localN)
        
        (xs, xe), = self.dax.getRanges()
        sw        = self.dax.getStencilWidth()
        
        for i in range(xs, xe):
            ix = i-xs+sw
            iy = i-xs
            
            y[iy] = (1. * x[ix-2] - 16. * x[ix-1] + 30. * x[ix] - 16. * x[ix+1] + 1. * x[ix+2]) * self.hx2_inv / 12. \
                  + (n[ix] - nmean) * self.charge            
        
    
