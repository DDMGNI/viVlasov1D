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
        
        assert dax.getDim() == 1
        assert da1.getDim() == 2
        assert da2.getDim() == 2
        
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
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # poisson constant
        self.poisson_const = poisson_const
        
        # collision parameter
        self.alpha = alpha
        
        # create working vectors
        self.VF  = self.da1.createGlobalVec()
        
        # create local vectors
        self.localB   = da2.createLocalVec()
        self.localF   = da1.createLocalVec()
        self.localH0  = da1.createLocalVec()
        self.localH1  = da1.createLocalVec()
        self.localVF  = da1.createLocalVec()

        # create Arakawa solver object
        self.arakawa     = PETScArakawa(da1, nx, nv, hx, hv)
        
    
    @cython.boundscheck(False)
    def calculate_moments(self, Vec F):
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] gf = self.da1.getVecArray(F)[...]
        cdef np.ndarray[np.float64_t, ndim=2] vf = self.da1.getVecArray(self.VF)[...]
        
        for j in np.arange(0, ye-ys):
            vf[:, j] = gf[:, j] * self.v[j]
        
        
    
#    @cython.boundscheck(False)
    def formMat(self, Mat A, Vec F, Vec H1):
        cdef np.int64_t i, j, ix, jx, tj
        cdef np.int64_t xe, xs, ye, ys
        
        cdef np.float64_t integral
        
        self.calculate_moments(F)
        
        self.da1.globalToLocal(F,       self.localF)
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da1.globalToLocal(H1,      self.localH1)
        self.da1.globalToLocal(self.VF, self.localVF)

        cdef np.ndarray[np.float64_t, ndim=2] fh = self.da1.getVecArray(self.localF) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1 = self.da1.getVecArray(self.localH1)[...]
        cdef np.ndarray[np.float64_t, ndim=2] vf = self.da1.getVecArray(self.localVF)[...]
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        
        cdef np.float64_t time_fac = 1.0 / (16. * self.ht)
        cdef np.float64_t arak_fac = 0.5 / (12. * self.hx * self.hv)
        cdef np.float64_t poss_fac = 0.25 * self.hv * self.poisson_const
         
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            for j in np.arange(ys, ye):
                jx = j-ys
                
                row.index = (i,j)
                
                # f
                row.field = 0
                
                if j == 0 or j == self.nv-1:
                    A.setValueStencil(row, row, 1.0, addv=PETSc.InsertMode.ADD_VALUES)
#                    self.setValueStencil(A.mat, row, row, 1.0)
                
                else:
                    for index, value in [
                            ((i-1, j-1), 1. * time_fac - (h0[ix-1, jx  ] - h0[ix,   jx-1]) * arak_fac \
                                                       - (h1[ix-1, jx  ] - h1[ix,   jx-1]) * arak_fac),
                            ((i-1, j  ), 2. * time_fac - (h0[ix,   jx+1] - h0[ix,   jx-1]) * arak_fac \
                                                       - (h0[ix-1, jx+1] - h0[ix-1, jx-1]) * arak_fac \
                                                       - (h1[ix,   jx+1] - h1[ix,   jx-1]) * arak_fac \
                                                       - (h1[ix-1, jx+1] - h1[ix-1, jx-1]) * arak_fac),
                            ((i-1, j+1), 1. * time_fac - (h0[ix,   jx+1] - h0[ix-1, jx  ]) * arak_fac \
                                                       - (h1[ix,   jx+1] - h1[ix-1, jx  ]) * arak_fac),
                            ((i,   j-1), 2. * time_fac + (h0[ix+1, jx  ] - h0[ix-1, jx  ]) * arak_fac \
                                                       + (h0[ix+1, jx-1] - h0[ix-1, jx-1]) * arak_fac \
                                                       + (h1[ix+1, jx  ] - h1[ix-1, jx  ]) * arak_fac \
                                                       + (h1[ix+1, jx-1] - h1[ix-1, jx-1]) * arak_fac),
                            ((i,   j  ), 4. * time_fac),
                            ((i,   j+1), 2. * time_fac - (h0[ix+1, jx  ] - h0[ix-1, jx  ]) * arak_fac \
                                                       - (h0[ix+1, jx+1] - h0[ix-1, jx+1]) * arak_fac \
                                                       - (h1[ix+1, jx  ] - h1[ix-1, jx  ]) * arak_fac \
                                                       - (h1[ix+1, jx+1] - h1[ix-1, jx+1]) * arak_fac),
                            ((i+1, j-1), 1. * time_fac + (h0[ix+1, jx  ] - h0[ix,   jx-1]) * arak_fac \
                                                       + (h1[ix+1, jx  ] - h1[ix,   jx-1]) * arak_fac),
                            ((i+1, j  ), 2. * time_fac + (h0[ix,   jx+1] - h0[ix,   jx-1]) * arak_fac \
                                                       + (h0[ix+1, jx+1] - h0[ix+1, jx-1]) * arak_fac \
                                                       + (h1[ix,   jx+1] - h1[ix,   jx-1]) * arak_fac \
                                                       + (h1[ix+1, jx+1] - h1[ix+1, jx-1]) * arak_fac),
                            ((i+1, j+1), 1. * time_fac + (h0[ix,   jx+1] - h0[ix+1, jx  ]) * arak_fac \
                                                       + (h1[ix,   jx+1] - h1[ix+1, jx  ]) * arak_fac),
                        ]:
                        
                        col.index = index
                        col.field = 0
                        A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                        
                                
                    for index, value in [
                            ((i-1, j-1), + (fh[ix-1, jx  ] - fh[ix,   jx-1]) * arak_fac),
                            ((i-1, j  ), + (fh[ix,   jx+1] - fh[ix,   jx-1]) * arak_fac \
                                         + (fh[ix-1, jx+1] - fh[ix-1, jx-1]) * arak_fac),
                            ((i-1, j+1), + (fh[ix,   jx+1] - fh[ix-1, jx  ]) * arak_fac),
                            ((i,   j-1), - (fh[ix+1, jx  ] - fh[ix-1, jx  ]) * arak_fac \
                                         - (fh[ix+1, jx-1] - fh[ix-1, jx-1]) * arak_fac),
                            ((i,   j+1), + (fh[ix+1, jx  ] - fh[ix-1, jx  ]) * arak_fac \
                                         + (fh[ix+1, jx+1] - fh[ix-1, jx+1]) * arak_fac),
                            ((i+1, j-1), - (fh[ix+1, jx  ] - fh[ix,   jx-1]) * arak_fac),
                            ((i+1, j  ), - (fh[ix,   jx+1] - fh[ix,   jx-1]) * arak_fac \
                                         - (fh[ix+1, jx+1] - fh[ix+1, jx-1]) * arak_fac),
                            ((i+1, j+1), - (fh[ix,   jx+1] - fh[ix+1, jx  ]) * arak_fac),
                        ]:
                        
                        col.index = index
                        col.field = 1
                        A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                        
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
#                        jpp = + x[i+1, j  ] * (h[i,   j+1] - h[i,   j-1]) \
#                              - x[i-1, j  ] * (h[i,   j+1] - h[i,   j-1]) \
#                              - x[i,   j+1] * (h[i+1, j  ] - h[i-1, j  ]) \
#                              + x[i,   j-1] * (h[i+1, j  ] - h[i-1, j  ])
#                        
#                        jpc = + x[i+1, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
#                              - x[i-1, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
#                              - x[i,   j+1] * (h[i+1, j+1] - h[i-1, j+1]) \
#                              + x[i,   j-1] * (h[i+1, j-1] - h[i-1, j-1])
#                        
#                        jcp = + x[i+1, j+1] * (h[i,   j+1] - h[i+1, j  ]) \
#                              - x[i-1, j-1] * (h[i-1, j  ] - h[i,   j-1]) \
#                              - x[i-1, j+1] * (h[i,   j+1] - h[i-1, j  ]) \
#                              + x[i+1, j-1] * (h[i+1, j  ] - h[i,   j-1])
#                        
#                        result = (jpp + jpc + jcp) / (12. * self.hx * self.hv)
#                        
                
                
                # phi
                row.field = 1
                
                for tj in np.arange(0, self.nv):
                    
#                    print(i,j,tj)
                    
                    for index, value in [
                            ((i-1, tj), 1. * poss_fac),
                            ((i,   tj), 2. * poss_fac),
                            ((i+1, tj), 1. * poss_fac),
                        ]:
                        
                        col.index = index
                        col.field = 0
                        A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                
                
                for index, value in [
                        ((i-1, j), -1. * self.hx2_inv),
                        ((i,   j), +2. * self.hx2_inv),
                        ((i+1, j), -1. * self.hx2_inv),
                    ]:
                    
                    col.index = index
                    col.field = 1
                    A.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                
        A.assemble()
        
        
        
#    @cython.boundscheck(False)
    def formRHS(self, Vec B, Vec F, Vec H1):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        cdef np.float64_t fsum
        
        self.calculate_moments(F)
        
        self.da1.globalToLocal(F,       self.localF)
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da1.globalToLocal(H1,      self.localH1)
        self.da1.globalToLocal(self.VF, self.localVF)
        
        cdef np.ndarray[np.float64_t, ndim=3] b  = self.da2.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh = self.da1.getVecArray(self.localF) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1 = self.da1.getVecArray(self.localH1)[...]
        cdef np.ndarray[np.float64_t, ndim=2] vf = self.da1.getVecArray(self.localVF)[...]
        
        
        fsum = F.sum() * self.hv / self.nx
        
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            b[iy, :, 1] = self.poisson_const * fsum
            
            
            # Vlasov equation
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
                
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[iy, jy, 0] = 0.0
                    
                else:
                    b[iy, jy, 0] = self.time_derivative(fh, ix, jx) \
                                 - 0.5 * self.arakawa.arakawa(fh, h0, ix, jx)
    


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
    
    
#    cdef setValueStencil(self, PetscMat matrix, PetscMatStencil row, PetscMatStencil col, PetscScalar value):
#        cdef _Mat_Stencil r = row, c = col
#        cdef PetscInsertMode im = insertmode(None)
#        matsetvaluestencil(matrix, r, c, value, im, 0)
    
