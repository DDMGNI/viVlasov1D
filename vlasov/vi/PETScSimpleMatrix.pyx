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
    
    def __init__(self, DA da1, DA da2, DA dax, Vec H0,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 np.float64_t poisson_const, np.float64_t alpha=0.):
        '''
        Constructor
        '''
        
        # distributed array
        self.dax = dax
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
        
        # create moment vectors
        self.A1 = self.dax.createGlobalVec()
        self.A2 = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1  = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localF   = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()

        self.localA1 = dax.createLocalVec()
        self.localA2 = dax.createLocalVec()
        
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
                        ((i-1,), - 1. * self.hx2_inv),
                        ((i,  ), + 2. * self.hx2_inv),
                        ((i+1,), - 1. * self.hx2_inv),
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
                        
                    for index, field, value in [
                            ((i-1,), j-1, 1. * time_fac - (h[ix-1, j  ] - h[ix,   j-1]) * arak_fac),
                            ((i-1,), j  , 2. * time_fac - (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
                                                        - (h[ix-1, j+1] - h[ix-1, j-1]) * arak_fac),
                            ((i-1,), j+1, 1. * time_fac - (h[ix,   j+1] - h[ix-1, j  ]) * arak_fac),
                            ((i,  ), j-1, 2. * time_fac + (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
                                                        + (h[ix+1, j-1] - h[ix-1, j-1]) * arak_fac),
                            ((i,  ), j  , 4. * time_fac),
                            ((i,  ), j+1, 2. * time_fac - (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
                                                        - (h[ix+1, j+1] - h[ix-1, j+1]) * arak_fac),
                            ((i+1,), j-1, 1. * time_fac + (h[ix+1, j  ] - h[ix,   j-1]) * arak_fac),
                            ((i+1,), j  , 2. * time_fac + (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
                                                        + (h[ix+1, j+1] - h[ix+1, j-1]) * arak_fac),
                            ((i+1,), j+1, 1. * time_fac + (h[ix,   j+1] - h[ix+1, j  ]) * arak_fac),
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
                
        A.assemble()
        
        
        
    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.int64_t i, j, ix, iy, xs, xe
        
        cdef np.float64_t fsum = self.Fh.sum() * self.hv / self.nx
        
        self.da1.globalToLocal(self.H0,  self.localH0)
#        self.da1.globalToLocal(self.H1h, self.localH1h)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        
        cdef np.ndarray[np.float64_t, ndim=2] b   = self.da2.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0 )[...]
#        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        
        
        (xs, xe), = self.da2.getRanges()
        
        
        # calculate moments
        cdef np.ndarray[np.float64_t, ndim=1] A1 = self.dax.getVecArray(self.A1)[...]
        cdef np.ndarray[np.float64_t, ndim=1] A2 = self.dax.getVecArray(self.A2)[...]

        cdef np.ndarray[np.float64_t, ndim=1] mom_n = np.zeros_like(A1)         # density
        cdef np.ndarray[np.float64_t, ndim=1] mom_u = np.zeros_like(A1)         # mean velocity
        cdef np.ndarray[np.float64_t, ndim=1] mom_e = np.zeros_like(A1)         # energy
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            mom_n[iy] = 0.
            mom_u[iy] = 0.
            mom_e[iy] = 0.
            
            for j in np.arange(0, (self.nv-1)/2):
                mom_n[iy] += fh[ix, j] + fh[ix, self.nv-1-j]
                mom_u[iy] += self.v[j]    * fh[ix, j] + self.v[self.nv-1-j]    * fh[ix, self.nv-1-j]
                mom_e[iy] += self.v[j]**2 * fh[ix, j] + self.v[self.nv-1-j]**2 * fh[ix, self.nv-1-j]

            mom_n[iy] += fh[ix, (self.nv-1)/2]
            mom_u[iy] += self.v[(self.nv-1)/2]    * fh[ix, (self.nv-1)/2]
            mom_e[iy] += self.v[(self.nv-1)/2]**2 * fh[ix, (self.nv-1)/2]
                
            mom_n[iy] *= self.hv
            mom_u[iy] *= self.hv / mom_n[iy]
            mom_e[iy] *= self.hv / mom_n[iy]
            
            A1[iy] = mom_u[iy]
            A2[iy] = 1. / ( mom_u[iy]**2 - mom_e[iy] )
        
        
        self.dax.globalToLocal(self.A1, self.localA1)
        self.dax.globalToLocal(self.A2, self.localA2)
        
        A1 = self.dax.getVecArray(self.localA1)[...]
        A2 = self.dax.getVecArray(self.localA2)[...]
        
        
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
                             + self.alpha * self.coll1(fh, A1, A2, ix, j) \
                             + self.alpha * self.coll2(fh, ix, j)
    


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
    cdef np.float64_t coll1(self, np.ndarray[np.float64_t, ndim=2] f,
                                  np.ndarray[np.float64_t, ndim=1] A1,
                                  np.ndarray[np.float64_t, ndim=1] A2,
                                  np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
        
        cdef np.float64_t result
        
        # d/dv ( v * A2 * f )
        result = 0.25 * ( \
                          + 1. * ( (A1[i-1] - v[j+1]) * f[i-1, j+1] - (A1[i-1] - v[j-1]) * f[i-1, j-1] ) * A2[i-1] \
                          + 2. * ( (A1[i  ] - v[j+1]) * f[i,   j+1] - (A1[i  ] - v[j-1]) * f[i,   j-1] ) * A2[i  ] \
                          + 1. * ( (A1[i+1] - v[j+1]) * f[i+1, j+1] - (A1[i+1] - v[j-1]) * f[i+1, j-1] ) * A2[i+1] \
                        ) * 0.5 / self.hv
        
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t coll2(self, np.ndarray[np.float64_t, ndim=2] f,
                                  np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * ( f[i-1, j+1] - 2. * f[i-1, j  ] + f[i-1, j-1] ) \
                     + 2. * ( f[i,   j+1] - 2. * f[i,   j  ] + f[i,   j-1] ) \
                     + 1. * ( f[i+1, j+1] - 2. * f[i+1, j  ] + f[i+1, j-1] ) \
                 ) * 0.25 * self.hv2_inv
        
        return result
    
