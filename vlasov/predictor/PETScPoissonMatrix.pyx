'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport DA, Mat, Vec#, PetscMat, PetscScalar

from vlasov.predictor.PETScArakawa import PETScArakawa


cdef class PETScPoissonMatrix(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DA da1, DA dax,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t hx, np.float64_t hv,
                 np.float64_t poisson_const, np.float64_t eps=0.):
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
        
        self.hx2     = hx**2
        self.hx2_inv = 1. / self.hx2 
        
        # poisson constant
        self.poisson_const = poisson_const
        self.eps = eps
        
        # create local vectors
        self.localX = dax.createLocalVec()
        self.localF = da1.createLocalVec()

        
    
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
            
            for index, value in [
                    ((i-1,), self.eps - 1. * self.hx2_inv),
                    ((i,  ), self.eps + 2. * self.hx2_inv),
                    ((i+1,), self.eps - 1. * self.hx2_inv),
                ]:
                
                col.index = index
                col.field = 0
                A.setValueStencil(row, col, value)
            
        
        A.assemble()
        
        
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
            ix = i-xs+1
            iy = i-xs
            
            integral = ( \
                         + 1. * f[ix-1, :].sum() \
                         + 2. * f[ix,   :].sum() \
                         + 1. * f[ix+1, :].sum() \
                       ) * 0.25 * self.hv
            
            b[iy] = - (integral - fsum) * self.poisson_const
        
