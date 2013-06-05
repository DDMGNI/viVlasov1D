'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport Mat, Vec  # , PetscMat, PetscScalar


cdef class PETScPoissonMatrix(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, VIDA da1, VIDA dax,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t hx, np.float64_t hv,
                 np.float64_t charge):
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
        
        self.hx2 = hx**2
        self.hx2_inv = 1. / self.hx2 
        
        # poisson constant
        self.charge = charge
        
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
            
#             for index, value in [
#                     ((i-2,), + 0.25 * self.hx2_inv),
#                     ((i-1,), - 2.   * self.hx2_inv),
#                     ((i,  ), + 3.5  * self.hx2_inv),
#                     ((i+1,), - 2.   * self.hx2_inv),
#                     ((i+2,), + 0.25 * self.hx2_inv),
#                 ]:
            for index, value in [
                    ((i-1,), - 1. * self.hx2_inv),
                    ((i,  ), + 2. * self.hx2_inv),
                    ((i+1,), - 1. * self.hx2_inv),
                ]:
#             for index, value in [
#                     ((i-2,), + 1. * self.hx2_inv / 6.),
#                     ((i-1,), + 2. * self.hx2_inv / 6.),
#                     ((i,  ), - 6. * self.hx2_inv / 6.),
#                     ((i+1,), + 2. * self.hx2_inv / 6.),
#                     ((i+2,), + 1. * self.hx2_inv / 6.),
#                 ]:
#             for index, value in [
#                     ((i-2,), +  1. * self.hx2_inv / 12.),
#                     ((i-1,), - 16. * self.hx2_inv / 12.),
#                     ((i,  ), + 30. * self.hx2_inv / 12.),
#                     ((i+1,), - 16. * self.hx2_inv / 12.),
#                     ((i+2,), +  1. * self.hx2_inv / 12.),
#                 ]:
                
                col.index = index
                A.setValueStencil(row, col, value)
            
        A.assemble()
        
    
    @cython.boundscheck(False)
    def formRHS(self, Vec N, Vec B):
        cdef np.int64_t i, ix, iy
        cdef np.int64_t xs, xe
        
        cdef np.float64_t nmean = N.sum() / self.nx
        
        cdef np.ndarray[np.float64_t, ndim = 1] b = self.dax.getGlobalArray(B)
        cdef np.ndarray[np.float64_t, ndim = 1] n = self.dax.getLocalArray(N, self.localN)
        
        
        (xs, xe), = self.dax.getRanges()
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            b[iy] = - ( n[ix] - nmean) * self.charge
#             b[iy] = - ( ( n[ix-2] + 8. * n[ix-1] + 18. * n[ix] + 8. * n[ix+1] + n[ix+2] ) / 36. - nmean) * self.charge
