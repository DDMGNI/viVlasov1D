'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport DA, Mat, Vec  # , PetscMat, PetscScalar

from vlasov.predictor.PETScArakawa import PETScArakawa


cdef class PETScPoissonMatrix(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DA da1, DA dax,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t hx, np.float64_t hv,
                 np.float64_t poisson_const):
        '''
        Constructor
        '''
        
        # distributed array
        self.dax = dax
        self.da1 = da1
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.hx = hx
        self.hv = hv
        
        self.hx2 = hx ** 2
        self.hx2_inv = 1. / self.hx2 
        
        # poisson constant
        self.poisson_const = poisson_const
        
        # create local vectors
        self.localX = dax.createLocalVec()
        self.localN = dax.createLocalVec()

        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j
        cdef np.int64_t xe, xs
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        (xs, xe), = self.dax.getRanges()
        
        
        # Laplace operator
        for i in np.arange(xs, xe):
            row.index = (i,)
            row.field = 0
            col.field = 0
            
            if i == 0:
                A.setValueStencil(row, row, 1.)
                
#                for j in np.arange(0, self.nx):
#                    col.index = (j,)
#                    A.setValueStencil(row, col, 1.)
            
            else:
                for index, value in [
                        ((i-1,), -1. * self.hx2_inv),
                        ((i,  ), +2. * self.hx2_inv),
                        ((i+1,), -1. * self.hx2_inv),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
            
        
        A.assemble()
        
    
    @cython.boundscheck(False)
    def formRHS(self, Vec N, Vec B):
        cdef np.int64_t i, ix, iy
        cdef np.int64_t xs, xe
        
        cdef np.float64_t nmean = N.sum() / self.nx
        
        self.dax.globalToLocal(N, self.localN)
        
        cdef np.ndarray[np.float64_t, ndim = 1] b = self.dax.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim = 1] n = self.dax.getVecArray(self.localN)[...]
        
        
        (xs, xe), = self.dax.getRanges()
        
        for i in np.arange(xs, xe):
            ix = i - xs + 1
            iy = i - xs
            
            if i == 0:
                b[iy] = 0.
                
            else:
                b[iy] = - ( 0.25 * ( n[ix-1] + 2. * n[ix  ] + n[ix+1] ) - nmean) * self.poisson_const
        
