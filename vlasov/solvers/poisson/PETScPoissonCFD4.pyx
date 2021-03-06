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
                    ((i-2,), +  5. * self.hx2_inv / 12.),
                    ((i-1,), - 32. * self.hx2_inv / 12.),
                    ((i,  ), + 54. * self.hx2_inv / 12.),
                    ((i+1,), - 32. * self.hx2_inv / 12.),
                    ((i+2,), +  5. * self.hx2_inv / 12.),
                ]:
                
                col.index = index
                A.setValueStencil(row, col, value)
            
        
        A.assemble()
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef int i, ix, iy
        cdef int xs, xe, sw
        
        cdef double[:] y = getGlobalArray(self.dax, Y)
        cdef double[:] x = getLocalArray(self.dax, X, self.localX)
        
        (xs, xe), = self.dax.getRanges()
        sw        = self.dax.getStencilWidth()
        
        for i in range(xs, xe):
            ix = i-xs+sw
            iy = i-xs
        
            y[iy] = (5. * x[ix-2] - 32. * x[ix-1] + 54. * x[ix] - 32. * x[ix+1] + 5. * x[ix+2]) * self.hx2_inv / 12.
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function_mult(self, Vec X, Vec N, Vec Y):
        cdef int i, ix, iy
        cdef int xs, xe, sw
        
        cdef double nmean = N.sum() / self.nx
        
        cdef double[:] y = getGlobalArray(self.dax, Y)
        cdef double[:] x = getLocalArray(self.dax, X, self.localX)
        cdef double[:] n = getLocalArray(self.dax, N, self.localN)
        
        (xs, xe), = self.dax.getRanges()
        sw        = self.dax.getStencilWidth()
        
        for i in range(xs, xe):
            ix = i-xs+sw
            iy = i-xs
        
            y[iy] = (5. * x[ix-2] - 32. * x[ix-1] + 54. * x[ix] - 32. * x[ix+1] + 5. * x[ix+2]) * self.hx2_inv / 12. \
                  + (n[ix] - nmean) * self.charge            
        
    
