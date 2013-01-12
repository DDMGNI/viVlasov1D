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


cdef class PETScMatrix(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DA da1, DA da2, DA dax, DA day, Vec H0,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 np.float64_t poisson_const, np.float64_t alpha=0.):
        '''
        Constructor
        '''
        
        # distributed array
        self.dax = dax
        self.day = day
        self.da1 = da1
        self.da2 = da2
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv
        
        self.hx2     = hx**2
        self.hx2_inv = 1. / self.hx2 
        
        self.hv2     = hv**2
        self.hv2_inv = 1. / self.hv2 
        
        # velocity grid
        self.v = v.copy()
        
        # poisson constant
        self.poisson_const = poisson_const
        
        # collision parameter
        self.alpha = alpha
        
        # create work and history vectors
        self.H0  = self.da1.createGlobalVec()
        self.H1  = self.da1.createGlobalVec()
        self.H1h = self.da1.createGlobalVec()
        self.F   = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1  = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localF   = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()

        # kinetic Hamiltonian
        H0.copy(self.H0)
        
        # create Arakawa solver object
        self.arakawa = PETScArakawa(da1, nx, nv, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1):
        F.copy(self.Fh)
        H1.copy(self.H1h)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j, ix
        cdef np.int64_t xe, xs
        
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.H1h, self.localH1h)

        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1h
        
        cdef np.float64_t time_fac = 1.0 / (16. * self.ht)
        cdef np.float64_t arak_fac = 0.5 / (12. * self.hx * self.hv)
        cdef np.float64_t poss_fac = 0.25 * self.hv * self.poisson_const
        cdef np.float64_t dvdv_fac = 0.5 * self.alpha * 0.25 * self.hv2_inv
        cdef np.float64_t f_fac    = 0.5 * self.alpha / 16.
        cdef np.float64_t coll_fac = 0.5 * self.alpha * 0.25 * 0.25 / self.hv
#        cdef np.float64_t f_fac    = 1.0 * self.alpha / 16.
#        cdef np.float64_t coll_fac = 1.0 * self.alpha * 0.25 * 0.25 / self.hv
#        cdef np.float64_t dvdv_fac = 1.0 * self.alpha * 0.25 * self.hv2_inv
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        (xs, xe), = self.da2.getRanges()
        
        
        # Poisson equation
        for i in np.arange(xs, xe):
            row.index = (i,)
            row.field = self.nv
            
            
            if i == 0:
                col.index = (i,)
                col.field = self.nv
                value     = 1.
                A.setValueStencil(row, col, value)
                
            else:
                # density: velocity integral of f
                for index, value in [
                        ((i-1,), 1. * poss_fac),
                        ((i,  ), 2. * poss_fac),
                        ((i+1,), 1. * poss_fac),
                    ]:
                        
                    col.index = index
                    
                    for j in np.arange(0, self.nv):
                        col.field = j
                        A.setValueStencil(row, col, value)
                
                # Laplace operator
                for index, value in [
                        ((i-1,), self.eps - 1. * self.hx2_inv),
                        ((i,  ), self.eps + 2. * self.hx2_inv),
                        ((i+1,), self.eps - 1. * self.hx2_inv),
                    ]:
                    
                    col.index = index
                    col.field = self.nv
                    A.setValueStencil(row, col, value)
                    
            
        for i in np.arange(xs, xe):
            ix = i-xs+1
            
            row.index = (i,)
                
            # Vlasov equation
            for j in np.arange(0, self.nv):
                row.field = j
                
                # Dirichlet boundary conditions
                if j == 0 or j == self.nv-1:
                    A.setValueStencil(row, row, 1.0)
                    
                else:
                    
#                    for index, field, value in [
#                            ((i-1,), j-1, 1. * time_fac),
#                            ((i-1,), j  , 2. * time_fac),
#                            ((i-1,), j+1, 1. * time_fac),
#                            ((i,  ), j-1, 2. * time_fac),
#                            ((i,  ), j  , 4. * time_fac),
#                            ((i,  ), j+1, 2. * time_fac),
#                            ((i+1,), j-1, 1. * time_fac),
#                            ((i+1,), j  , 2. * time_fac),
#                            ((i+1,), j+1, 1. * time_fac),
#                        ]:
#                                                
#                    for index, field, value in [
#                            ((i-1,), j-1, 1. * time_fac - (h0 [ix-1, j  ] - h0 [ix,   j-1]) * arak_fac),
#                            ((i-1,), j  , 2. * time_fac - (h0 [ix,   j+1] - h0 [ix,   j-1]) * arak_fac \
#                                                        - (h0 [ix-1, j+1] - h0 [ix-1, j-1]) * arak_fac),
#                            ((i-1,), j+1, 1. * time_fac - (h0 [ix,   j+1] - h0 [ix-1, j  ]) * arak_fac),
#                            ((i,  ), j-1, 2. * time_fac + (h0 [ix+1, j  ] - h0 [ix-1, j  ]) * arak_fac \
#                                                        + (h0 [ix+1, j-1] - h0 [ix-1, j-1]) * arak_fac),
#                            ((i,  ), j  , 4. * time_fac),
#                            ((i,  ), j+1, 2. * time_fac - (h0 [ix+1, j  ] - h0 [ix-1, j  ]) * arak_fac \
#                                                        - (h0 [ix+1, j+1] - h0 [ix-1, j+1]) * arak_fac),
#                            ((i+1,), j-1, 1. * time_fac + (h0 [ix+1, j  ] - h0 [ix,   j-1]) * arak_fac),
#                            ((i+1,), j  , 2. * time_fac + (h0 [ix,   j+1] - h0 [ix,   j-1]) * arak_fac \
#                                                        + (h0 [ix+1, j+1] - h0 [ix+1, j-1]) * arak_fac),
#                            ((i+1,), j+1, 1. * time_fac + (h0 [ix,   j+1] - h0 [ix+1, j  ]) * arak_fac),
#                        ]:
#
#                    for index, field, value in [
#                            ((i-1,), j-1, 1. * time_fac - (h[ix-1, j  ] - h[ix,   j-1]) * arak_fac),
#                            ((i-1,), j  , 2. * time_fac - (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
#                                                        - (h[ix-1, j+1] - h[ix-1, j-1]) * arak_fac),
#                            ((i-1,), j+1, 1. * time_fac - (h[ix,   j+1] - h[ix-1, j  ]) * arak_fac),
#                            ((i,  ), j-1, 2. * time_fac + (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
#                                                        + (h[ix+1, j-1] - h[ix-1, j-1]) * arak_fac),
#                            ((i,  ), j  , 4. * time_fac),
#                            ((i,  ), j+1, 2. * time_fac - (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
#                                                        - (h[ix+1, j+1] - h[ix-1, j+1]) * arak_fac),
#                            ((i+1,), j-1, 1. * time_fac + (h[ix+1, j  ] - h[ix,   j-1]) * arak_fac),
#                            ((i+1,), j  , 2. * time_fac + (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
#                                                        + (h[ix+1, j+1] - h[ix+1, j-1]) * arak_fac),
#                            ((i+1,), j+1, 1. * time_fac + (h[ix,   j+1] - h[ix+1, j  ]) * arak_fac),
#                            ((i-1,), self.nv,    + 2. * (fh[ix,   j+1] - fh[ix,   j-1]) * arak_fac \
#                                                 + 1. * (fh[ix-1, j+1] - fh[ix-1, j-1]) * arak_fac),
#                            ((i,  ), self.nv,    + 1. * (fh[ix-1, j-1] - fh[ix+1, j-1]) * arak_fac \
#                                                 + 1. * (fh[ix+1, j+1] - fh[ix-1, j+1]) * arak_fac),
#                            ((i+1,), self.nv,    + 2. * (fh[ix,   j-1] - fh[ix,   j+1]) * arak_fac \
#                                                 + 1. * (fh[ix+1, j-1] - fh[ix+1, j+1]) * arak_fac),
#                        ]:
                        
                        
#                    for index, field, value in [
#                            ((i-1,), j-1, 1. * time_fac - (h[ix-1, j  ] - h[ix,   j-1]) * arak_fac \
#                                                        - 1. * dvdv_fac),
#                            ((i-1,), j  , 2. * time_fac - (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
#                                                        - (h[ix-1, j+1] - h[ix-1, j-1]) * arak_fac \
#                                                        + 2. * dvdv_fac),
#                            ((i-1,), j+1, 1. * time_fac - (h[ix,   j+1] - h[ix-1, j  ]) * arak_fac \
#                                                        - 1. * dvdv_fac),
#                            ((i,  ), j-1, 2. * time_fac + (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
#                                                        + (h[ix+1, j-1] - h[ix-1, j-1]) * arak_fac \
#                                                        - 2. * dvdv_fac),
#                            ((i,  ), j  , 4. * time_fac + 4. * dvdv_fac),
#                            ((i,  ), j+1, 2. * time_fac - (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
#                                                        - (h[ix+1, j+1] - h[ix-1, j+1]) * arak_fac \
#                                                        - 2. * dvdv_fac),
#                            ((i+1,), j-1, 1. * time_fac + (h[ix+1, j  ] - h[ix,   j-1]) * arak_fac \
#                                                        - 1. * dvdv_fac),
#                            ((i+1,), j  , 2. * time_fac + (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
#                                                        + (h[ix+1, j+1] - h[ix+1, j-1]) * arak_fac \
#                                                        + 2. * dvdv_fac),
#                            ((i+1,), j+1, 1. * time_fac + (h[ix,   j+1] - h[ix+1, j  ]) * arak_fac \
#                                                        - 1. * dvdv_fac),
#                            ((i-1,), self.nv,    + 2. * (fh[ix,   j+1] - fh[ix,   j-1]) * arak_fac \
#                                                 + 1. * (fh[ix-1, j+1] - fh[ix-1, j-1]) * arak_fac),
#                            ((i,  ), self.nv,    + 1. * (fh[ix-1, j-1] - fh[ix+1, j-1]) * arak_fac \
#                                                 + 1. * (fh[ix+1, j+1] - fh[ix-1, j+1]) * arak_fac),
#                            ((i+1,), self.nv,    + 2. * (fh[ix,   j-1] - fh[ix,   j+1]) * arak_fac \
#                                                 + 1. * (fh[ix+1, j-1] - fh[ix+1, j+1]) * arak_fac),
#                        ]:
                        
#                    for index, field, value in [
#                            ((i-1,), j-1, 1. * time_fac - (h[ix-1, j  ] - h[ix,   j-1]) * arak_fac \
#                                                        - 1. * f_fac \
#                                                        + 1. * coll_fac * (self.v[j-1] + self.v[j])),
#                            ((i-1,), j  , 2. * time_fac - (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
#                                                        - (h[ix-1, j+1] - h[ix-1, j-1]) * arak_fac \
#                                                        - 2. * f_fac \
#                                                        - 1. * coll_fac * (self.v[j-1] + self.v[j]) \
#                                                        + 1. * coll_fac * (self.v[j] + self.v[j+1])),
#                            ((i-1,), j+1, 1. * time_fac - (h[ix,   j+1] - h[ix-1, j  ]) * arak_fac \
#                                                        - 1. * f_fac \
#                                                        - 1. * coll_fac * (self.v[j] + self.v[j+1])),
#                            ((i,  ), j-1, 2. * time_fac + (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
#                                                        + (h[ix+1, j-1] - h[ix-1, j-1]) * arak_fac \
#                                                        - 2. * f_fac \
#                                                        + 2. * coll_fac * (self.v[j-1] + self.v[j])),
#                            ((i,  ), j  , 4. * time_fac \
#                                                        - 4. * f_fac \
#                                                        - 2. * coll_fac * (self.v[j-1] + self.v[j]) \
#                                                        + 2. * coll_fac * (self.v[j] + self.v[j+1])),
#                            ((i,  ), j+1, 2. * time_fac - (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
#                                                        - (h[ix+1, j+1] - h[ix-1, j+1]) * arak_fac \
#                                                        - 2. * f_fac \
#                                                        - 2. * coll_fac * (self.v[j] + self.v[j+1])),
#                            ((i+1,), j-1, 1. * time_fac + (h[ix+1, j  ] - h[ix,   j-1]) * arak_fac \
#                                                        - 1. * f_fac \
#                                                        + 1. * coll_fac * (self.v[j-1] + self.v[j])),
#                            ((i+1,), j  , 2. * time_fac + (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
#                                                        + (h[ix+1, j+1] - h[ix+1, j-1]) * arak_fac \
#                                                        - 2. * f_fac \
#                                                        - 1. * coll_fac * (self.v[j-1] + self.v[j]) \
#                                                        + 1. * coll_fac * (self.v[j] + self.v[j+1])),
#                            ((i+1,), j+1, 1. * time_fac + (h[ix,   j+1] - h[ix+1, j  ]) * arak_fac \
#                                                        - 1. * f_fac \
#                                                        - 1. * coll_fac * (self.v[j] + self.v[j+1])),
#                            ((i-1,), self.nv,    + 2. * (fh[ix,   j+1] - fh[ix,   j-1]) * arak_fac \
#                                                 + 1. * (fh[ix-1, j+1] - fh[ix-1, j-1]) * arak_fac),
#                            ((i,  ), self.nv,    + 1. * (fh[ix-1, j-1] - fh[ix+1, j-1]) * arak_fac \
#                                                 + 1. * (fh[ix+1, j+1] - fh[ix-1, j+1]) * arak_fac),
#                            ((i+1,), self.nv,    + 2. * (fh[ix,   j-1] - fh[ix,   j+1]) * arak_fac \
#                                                 + 1. * (fh[ix+1, j-1] - fh[ix+1, j+1]) * arak_fac),
#                        ]:
                        
                    for index, field, value in [
                            ((i-1,), j-1, 1. * time_fac - (h[ix-1, j  ] - h[ix,   j-1]) * arak_fac \
                                                        - 1. * dvdv_fac \
                                                        - 1. * f_fac \
                                                        + 1. * coll_fac * (self.v[j-1] + self.v[j])),
                            ((i-1,), j  , 2. * time_fac - (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
                                                        - (h[ix-1, j+1] - h[ix-1, j-1]) * arak_fac \
                                                        + 2. * dvdv_fac \
                                                        - 2. * f_fac \
                                                        - 1. * coll_fac * (self.v[j-1] + self.v[j]) \
                                                        + 1. * coll_fac * (self.v[j] + self.v[j+1])),
                            ((i-1,), j+1, 1. * time_fac - (h[ix,   j+1] - h[ix-1, j  ]) * arak_fac \
                                                        - 1. * dvdv_fac \
                                                        - 1. * f_fac \
                                                        - 1. * coll_fac * (self.v[j] + self.v[j+1])),
                            ((i,  ), j-1, 2. * time_fac + (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
                                                        + (h[ix+1, j-1] - h[ix-1, j-1]) * arak_fac \
                                                        - 2. * dvdv_fac \
                                                        - 2. * f_fac \
                                                        + 2. * coll_fac * (self.v[j-1] + self.v[j])),
                            ((i,  ), j  , 4. * time_fac + 4. * dvdv_fac \
                                                        - 4. * f_fac \
                                                        - 2. * coll_fac * (self.v[j-1] + self.v[j]) \
                                                        + 2. * coll_fac * (self.v[j] + self.v[j+1])),
                            ((i,  ), j+1, 2. * time_fac - (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
                                                        - (h[ix+1, j+1] - h[ix-1, j+1]) * arak_fac \
                                                        - 2. * dvdv_fac \
                                                        - 2. * f_fac \
                                                        - 2. * coll_fac * (self.v[j] + self.v[j+1])),
                            ((i+1,), j-1, 1. * time_fac + (h[ix+1, j  ] - h[ix,   j-1]) * arak_fac \
                                                        - 1. * dvdv_fac \
                                                        - 1. * f_fac \
                                                        + 1. * coll_fac * (self.v[j-1] + self.v[j])),
                            ((i+1,), j  , 2. * time_fac + (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
                                                        + (h[ix+1, j+1] - h[ix+1, j-1]) * arak_fac \
                                                        + 2. * dvdv_fac \
                                                        - 2. * f_fac \
                                                        - 1. * coll_fac * (self.v[j-1] + self.v[j]) \
                                                        + 1. * coll_fac * (self.v[j] + self.v[j+1])),
                            ((i+1,), j+1, 1. * time_fac + (h[ix,   j+1] - h[ix+1, j  ]) * arak_fac \
                                                        - 1. * dvdv_fac \
                                                        - 1. * f_fac \
                                                        - 1. * coll_fac * (self.v[j] + self.v[j+1])),
                            ((i-1,), self.nv,    + 2. * (fh[ix,   j+1] - fh[ix,   j-1]) * arak_fac \
                                                 + 1. * (fh[ix-1, j+1] - fh[ix-1, j-1]) * arak_fac),
                            ((i,  ), self.nv,    + 1. * (fh[ix-1, j-1] - fh[ix+1, j-1]) * arak_fac \
                                                 + 1. * (fh[ix+1, j+1] - fh[ix-1, j+1]) * arak_fac),
                            ((i+1,), self.nv,    + 2. * (fh[ix,   j-1] - fh[ix,   j+1]) * arak_fac \
                                                 + 1. * (fh[ix+1, j-1] - fh[ix+1, j+1]) * arak_fac),
                        ]:
                        
                        col.index = index
                        col.field = field
                        A.setValueStencil(row, col, value)

                    
#                        
#                        Time Derivative
#                        
#                        result = ( \
#                                   + 1. * x[i-1, j-1] \
#                                   + 2. * x[i-1, j  ] \
#                                   + 1. * x[i-1, j+1] \
#                                   + 2. * x[i,   j-1] \
#                                   + 4. * x[i,   j  ] \
#                                   + 2. * x[i,   j+1] \
#                                   + 1. * x[i+1, j-1] \
#                                   + 2. * x[i+1, j  ] \
#                                   + 1. * x[i+1, j+1] \
#                                 ) / (16. * self.ht)
#                        
#                        
#                        Arakawa Stencil
#                        
#                        jpp = + f[i+1, j  ] * (h[i,   j+1] - h[i,   j-1]) \
#                              - f[i-1, j  ] * (h[i,   j+1] - h[i,   j-1]) \
#                              - f[i,   j+1] * (h[i+1, j  ] - h[i-1, j  ]) \
#                              + f[i,   j-1] * (h[i+1, j  ] - h[i-1, j  ])
#                        
#                        jpc = + f[i+1, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
#                              - f[i-1, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
#                              - f[i,   j+1] * (h[i+1, j+1] - h[i-1, j+1]) \
#                              + f[i,   j-1] * (h[i+1, j-1] - h[i-1, j-1])
#                        
#                        jcp = + f[i+1, j+1] * (h[i,   j+1] - h[i+1, j  ]) \
#                              - f[i-1, j-1] * (h[i-1, j  ] - h[i,   j-1]) \
#                              - f[i-1, j+1] * (h[i,   j+1] - h[i-1, j  ]) \
#                              + f[i+1, j-1] * (h[i+1, j  ] - h[i,   j-1])
#                        
#                        result = (jpp + jpc + jcp) / (12. * self.hx * self.hv)
#                        
#                              + f[i+1, j  ] * h[i,   j+1] \
#                              - f[i+1, j  ] * h[i,   j-1] \
#                              - f[i-1, j  ] * h[i,   j+1] \
#                              + f[i-1, j  ] * h[i,   j-1] \
#                              - f[i,   j+1] * h[i+1, j  ] \
#                              + f[i,   j+1] * h[i-1, j  ] \
#                              + f[i,   j-1] * h[i+1, j  ] \
#                              - f[i,   j-1] * h[i-1, j  ] \
#                              + f[i+1, j  ] * h[i+1, j+1] \
#                              - f[i+1, j  ] * h[i+1, j-1] \
#                              - f[i-1, j  ] * h[i-1, j+1] \
#                              + f[i-1, j  ] * h[i-1, j-1] \
#                              - f[i,   j+1] * h[i+1, j+1] \
#                              + f[i,   j+1] * h[i-1, j+1] \
#                              + f[i,   j-1] * h[i+1, j-1] \
#                              - f[i,   j-1] * h[i-1, j-1] \
#                              + f[i+1, j+1] * h[i,   j+1] \
#                              - f[i+1, j+1] * h[i+1, j  ] \
#                              - f[i-1, j-1] * h[i-1, j  ] \
#                              + f[i-1, j-1] * h[i,   j-1] \
#                              - f[i-1, j+1] * h[i,   j+1] \
#                              + f[i-1, j+1] * h[i-1, j  ] \
#                              + f[i+1, j-1] * h[i+1, j  ] \
#                              - f[i+1, j-1] * h[i,   j-1] \
#                        
#                              + 2. * f[i,   j+1] * h[i-1] \
#                              - 2. * f[i,   j-1] * h[i-1] \
#                              - f[i-1, j-1] * h[i-1] \
#                              + f[i-1, j+1] * h[i-1] \
#                              + f[i+1, j+1] * h[i  ] \
#                              + f[i-1, j-1] * h[i  ] \
#                              - f[i-1, j+1] * h[i  ] \
#                              - f[i+1, j-1] * h[i  ] \
#                              - 2. * f[i,   j+1] * h[i+1] \
#                              + 2. * f[i,   j-1] * h[i+1] \
#                              - f[i+1, j+1] * h[i+1] \
#                              + f[i+1, j-1] * h[i+1] \
#                        
#                        res = - p[i+1] * (f[i,   j+1] - f[i,   j-1]) \
#                              - p[i+1] * (f[i,   j+1] - f[i+1, j  ]) \
#                              - p[i+1] * (f[i+1, j+1] - f[i+1, j-1]) \
#                              - p[i+1] * (f[i+1, j  ] - f[i,   j-1])
#                              + p[i  ] * (f[i+1, j+1] - f[i-1, j+1]) \
#                              - p[i  ] * (f[i+1, j-1] - f[i-1, j-1])
#                              + p[i-1] * (f[i,   j+1] - f[i,   j-1])
#                              + p[i-1] * (f[i,   j+1] - f[i-1, j  ]) \
#                              + p[i-1] * (f[i-1, j+1] - f[i-1, j-1]) \
#                              + p[i-1] * (f[i-1, j  ] - f[i,   j-1]) \
#                        
#                        result = res / (12. * self.hx * self.hv)
#                        
#                        
#                        dvdv
#                        
#                        result = ( \
#                                     + 1. * x[i-1, j-1] \
#                                     - 2. * x[i-1, j  ] \
#                                     + 1. * x[i-1, j+1] \
#                                     + 2. * x[i,   j-1] \
#                                     - 4. * x[i,   j  ] \
#                                     + 2. * x[i,   j+1] \
#                                     + 1. * x[i+1, j-1] \
#                                     - 2. * x[i+1, j  ] \
#                                     + 1. * x[i+1, j+1] \
#                                 ) * 0.25 * self.hv2_inv
#                        
#                        coll
#                
#                        result = ( \
#                                   + 1. * x[i-1, j+1] \
#                                   - 1. * x[i-1, j-1] \
#                                   + 2. * x[i,   j+1] \
#                                   - 2. * x[i,   j-1] \
#                                   + 1. * x[i+1, j+1] \
#                                   - 1. * x[i+1, j-1] \
#                                 ) * 0.25 / (2. * self.hv)
#                        
                
                
        A.assemble()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Matrix")
        
        
    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.int64_t i, j, ix, iy, xs, xe
        
        cdef np.float64_t fsum = self.Fh.sum() * self.hv / self.nx
        
        self.da1.globalToLocal(self.H0,  self.localH0)
#        self.da1.globalToLocal(self.H1h, self.localH1h)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.da1.globalToLocal(self.VFh, self.localVFh)
        
        cdef np.ndarray[np.float64_t, ndim=2] b   = self.da2.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0 )[...]
#        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        cdef np.ndarray[np.float64_t, ndim=2] vfh = self.da1.getVecArray(self.localVFh)[...]
        
        
        (xs, xe), = self.da2.getRanges()
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            if i == 0:
                b[iy, self.nv] = 0.
                
            else:
                b[iy, self.nv] = fsum * self.poisson_const
            
            
            # Vlasov equation
            for j in np.arange(0, self.nv):
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[iy, j] = 0.0
                    
                else:
                    b[iy, j] = self.time_derivative(fh, ix, j) \
                             - 0.5 * self.arakawa.arakawa(fh, h0, ix, j) \
                             + 0.5 * self.alpha * self.dvdv(fh, ix, j) \
                             + 0.5 * self.alpha * self.coll(fh, ix, j)
    
    
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     RHS")
        


    @cython.boundscheck(False)
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
                                            np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * x[i-1, j-1] \
                   + 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   + 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) / (16. * self.ht)
        
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t dvdv(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * x[i-1, j-1] \
                     - 2. * x[i-1, j  ] \
                     + 1. * x[i-1, j+1] \
                     + 2. * x[i,   j-1] \
                     - 4. * x[i,   j  ] \
                     + 2. * x[i,   j+1] \
                     + 1. * x[i+1, j-1] \
                     - 2. * x[i+1, j  ] \
                     + 1. * x[i+1, j+1] \
                 ) * 0.25 * self.hv2_inv
        
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t coll(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
#        result = ( \
#                   + 1. * ( x[i-1, j+1] - x[i-1, j-1] ) \
#                   + 2. * ( x[i,   j+1] - x[i,   j-1] ) \
#                   + 1. * ( x[i+1, j+1] - x[i+1, j-1] ) \
#                 ) * 0.25 / (2. * self.hv)
        
        result = ( \
                   + 1. * x[i-1, j-1] \
                   + 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   + 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) / 16. \
               + ( \
                   + 1. * ( x[i-1, j  ] - x[i-1, j-1] ) * (self.v[j-1] + self.v[j  ]) \
                   + 1. * ( x[i-1, j+1] - x[i-1, j  ] ) * (self.v[j  ] + self.v[j+1]) \
                   + 2. * ( x[i,   j  ] - x[i,   j-1] ) * (self.v[j-1] + self.v[j  ]) \
                   + 2. * ( x[i,   j+1] - x[i,   j  ] ) * (self.v[j  ] + self.v[j+1]) \
                   + 1. * ( x[i+1, j  ] - x[i+1, j-1] ) * (self.v[j-1] + self.v[j  ]) \
                   + 1. * ( x[i+1, j+1] - x[i+1, j  ] ) * (self.v[j  ] + self.v[j+1]) \
                 ) * 0.25 * 0.25 / self.hv
                 
        
        return result
    
    
