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
    
    def __init__(self, DA da1, DA dax, Vec H0,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 np.float64_t alpha=0.):
        '''
        Constructor
        '''
        
        assert dax.getDim() == 1
        assert da1.getDim() == 2
        
        # distributed array
        self.dax = dax
        self.da1 = da1
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv
        
        # velocity grid
        self.v = v.copy()
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # collision parameter
        self.alpha = alpha
        
        # create working vectors
        self.VF  = self.da1.createGlobalVec()
        
        # create local vectors
        self.localB   = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
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
    def formMat(self, Mat A, Vec H1):
        cdef np.int64_t i, j, ix, jx
        cdef np.int64_t xe, xs, ye, ys
        
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da1.globalToLocal(H1,      self.localH1)

        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1 = self.da1.getVecArray(self.localH1)[...]
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        cdef np.float64_t time_fac = 1.0 / (16. * self.ht)
        cdef np.float64_t arak_fac = 0.5 / (12. * self.hx * self.hv)
         
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            for j in np.arange(ys, ye):
                jx = j-ys
                
                row.index = (i,j)
                row.field = 0
                
                if j == 0 or j == self.nv-1:
                    A.setValueStencil(row, row, 1.0)
                
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
                
                
        A.assemble()
        
        
        
#    @cython.boundscheck(False)
    def formRHS(self, Vec B, Vec Fh, Vec H1):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
#        self.calculate_moments(Fh)
        
        self.da1.globalToLocal(Fh,      self.localFh)
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da1.globalToLocal(H1,      self.localH1)
        self.da1.globalToLocal(self.VF, self.localVF)
        
        cdef np.ndarray[np.float64_t, ndim=2] b  = self.da1.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh = self.da1.getVecArray(self.localFh)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1 = self.da1.getVecArray(self.localH1)[...]
        cdef np.ndarray[np.float64_t, ndim=2] vf = self.da1.getVecArray(self.localVF)[...]
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
                
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[iy, jy] = 0.0
                    
                else:
                    b[iy, jy] = self.time_derivative(fh, ix, jx) \
                              - 0.5 * self.arakawa.arakawa(fh, h0,  ix, jx) \
                              - 0.5 * self.arakawa.arakawa(fh, h1,  ix, jx)
    


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
