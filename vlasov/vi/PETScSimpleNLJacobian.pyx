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


cdef class PETScJacobian(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DA da1, DA da2, DA dax, Vec H0,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 np.float64_t poisson_const):
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
        
        # poisson constant
        self.poisson_const = poisson_const
        
        # create work and history vectors
        self.H0  = self.da1.createGlobalVec()
        self.H1p = self.da1.createGlobalVec()
        self.H1h = self.da1.createGlobalVec()
        self.Fp  = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1p = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localFp  = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()

        # kinetic Hamiltonian
        H0.copy(self.H0)
        
    
    def update_history(self, Vec F, Vec H1):
        F.copy(self.Fh)
        H1.copy(self.H1h)
        
    
    def update_previous(self, Vec X):
        (xs, xe), = self.da2.getRanges()
        
        H1 = self.da1.createGlobalVec()
        F  = self.da1.createGlobalVec()
        P  = self.dax.createGlobalVec()
        
        x  = self.da2.getVecArray(X)
        h1 = self.da1.getVecArray(H1)
        f  = self.da1.getVecArray(F)
        p  = self.dax.getVecArray(P)
        
        
        f[xs:xe] = x[xs:xe, 0:self.nv]
        p[xs:xe] = x[xs:xe,   self.nv]
        
        for j in np.arange(0, self.nv):
            h1[xs:xe, j] = p[xs:xe]
        
        
        F.copy(self.Fp)
        H1.copy(self.H1p)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j, ix
        cdef np.int64_t xe, xs
        
        self.da1.globalToLocal(self.Fp,  self.localFp)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.H1p, self.localH1p)
        self.da1.globalToLocal(self.H1h, self.localH1h)

        cdef np.ndarray[np.float64_t, ndim=2] fp  = self.da1.getVecArray(self.localFp) [...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h1p = self.da1.getVecArray(self.localH1p)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h_ave = h0 + 0.5 * (h1p + h1h)
#        cdef np.ndarray[np.float64_t, ndim=2] h_ave = h0 + h1p
        
        cdef np.float64_t time_fac = 1.0 / (16. * self.ht)
        cdef np.float64_t arak_fac = 0.5 / (12. * self.hx * self.hv)
        cdef np.float64_t poss_fac = 1.0 * 0.25 * self.hv * self.poisson_const
#        cdef np.float64_t poss_fac = 0.5 * 0.25 * self.hv * self.poisson_const
         
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
                
                A.setValueStencil(row, col, 1.)
                
#                for j in np.arange(0, self.nx):
#                    col.index = (j,)
#                    A.setValueStencil(row, col, 1.)
                
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
#                for index, value in [
#                        ((i-1,), - 0.5 * 1. * self.hx2_inv),
#                        ((i,  ), + 0.5 * 2. * self.hx2_inv),
#                        ((i+1,), - 0.5 * 1. * self.hx2_inv),
#                    ]:
                    
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
#                            ((i-1,), j-1, - (h_ave[ix-1, j  ] - h_ave[ix,   j-1]) * arak_fac),
#                            ((i-1,), j  , - (h_ave[ix,   j+1] - h_ave[ix,   j-1]) * arak_fac \
#                                          - (h_ave[ix-1, j+1] - h_ave[ix-1, j-1]) * arak_fac),
#                            ((i-1,), j+1, - (h_ave[ix,   j+1] - h_ave[ix-1, j  ]) * arak_fac),
#                            ((i,  ), j-1, + (h_ave[ix+1, j  ] - h_ave[ix-1, j  ]) * arak_fac \
#                                          + (h_ave[ix+1, j-1] - h_ave[ix-1, j-1]) * arak_fac),
#                            ((i,  ), j  , + 16. * time_fac),
#                            ((i,  ), j+1, - (h_ave[ix+1, j  ] - h_ave[ix-1, j  ]) * arak_fac \
#                                          - (h_ave[ix+1, j+1] - h_ave[ix-1, j+1]) * arak_fac),
#                            ((i+1,), j-1, + (h_ave[ix+1, j  ] - h_ave[ix,   j-1]) * arak_fac),
#                            ((i+1,), j  , + (h_ave[ix,   j+1] - h_ave[ix,   j-1]) * arak_fac \
#                                          + (h_ave[ix+1, j+1] - h_ave[ix+1, j-1]) * arak_fac),
#                            ((i+1,), j+1, + (h_ave[ix,   j+1] - h_ave[ix+1, j  ]) * arak_fac),
#                            ((i-1,), self.nv, + 2. * (f_ave[ix,   j+1] - f_ave[ix,   j-1]) * arak_fac \
#                                              + 1. * (f_ave[ix-1, j+1] - f_ave[ix-1, j-1]) * arak_fac),
#                            ((i,  ), self.nv, + 1. * (f_ave[ix-1, j-1] - f_ave[ix+1, j-1]) * arak_fac \
#                                              + 1. * (f_ave[ix+1, j+1] - f_ave[ix-1, j+1]) * arak_fac),
#                            ((i+1,), self.nv, + 2. * (f_ave[ix,   j-1] - f_ave[ix,   j+1]) * arak_fac \
#                                              + 1. * (f_ave[ix+1, j-1] - f_ave[ix+1, j+1]) * arak_fac),
#                        ]:
                        
                    for index, field, value in [
                            ((i-1,), j-1, 1. * time_fac - (h_ave[ix-1, j  ] - h_ave[ix,   j-1]) * arak_fac),
                            ((i-1,), j  , 2. * time_fac - (h_ave[ix,   j+1] - h_ave[ix,   j-1]) * arak_fac \
                                                        - (h_ave[ix-1, j+1] - h_ave[ix-1, j-1]) * arak_fac),
                            ((i-1,), j+1, 1. * time_fac - (h_ave[ix,   j+1] - h_ave[ix-1, j  ]) * arak_fac),
                            ((i,  ), j-1, 2. * time_fac + (h_ave[ix+1, j  ] - h_ave[ix-1, j  ]) * arak_fac \
                                                        + (h_ave[ix+1, j-1] - h_ave[ix-1, j-1]) * arak_fac),
                            ((i,  ), j  , 4. * time_fac),
                            ((i,  ), j+1, 2. * time_fac - (h_ave[ix+1, j  ] - h_ave[ix-1, j  ]) * arak_fac \
                                                        - (h_ave[ix+1, j+1] - h_ave[ix-1, j+1]) * arak_fac),
                            ((i+1,), j-1, 1. * time_fac + (h_ave[ix+1, j  ] - h_ave[ix,   j-1]) * arak_fac),
                            ((i+1,), j  , 2. * time_fac + (h_ave[ix,   j+1] - h_ave[ix,   j-1]) * arak_fac \
                                                        + (h_ave[ix+1, j+1] - h_ave[ix+1, j-1]) * arak_fac),
                            ((i+1,), j+1, 1. * time_fac + (h_ave[ix,   j+1] - h_ave[ix+1, j  ]) * arak_fac),
                            ((i-1,), self.nv,    + 2. * (f_ave[ix,   j+1] - f_ave[ix,   j-1]) * arak_fac \
                                                 + 1. * (f_ave[ix-1, j+1] - f_ave[ix-1, j-1]) * arak_fac),
                            ((i,  ), self.nv,    + 1. * (f_ave[ix-1, j-1] - f_ave[ix+1, j-1]) * arak_fac \
                                                 + 1. * (f_ave[ix+1, j+1] - f_ave[ix-1, j+1]) * arak_fac),
                            ((i+1,), self.nv,    + 2. * (f_ave[ix,   j-1] - f_ave[ix,   j+1]) * arak_fac \
                                                 + 1. * (f_ave[ix+1, j-1] - f_ave[ix+1, j+1]) * arak_fac),
                        ]:
                        
                        col.index = index
                        col.field = field
                        A.setValueStencil(row, col, value)

                    

        A.assemble()
        

