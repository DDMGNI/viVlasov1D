'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport Mat, Vec

from vlasov.Toolbox import Toolbox


cdef class PETScVlasovMatrix(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, VIDA da1, VIDA da2, VIDA dax, Vec H0,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 np.float64_t coll_freq=0.):
        '''
        Constructor
        '''
        
        # distributed array
        self.dax = dax
        self.da1 = da1
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv
        
        self.time_fac = 1.0 / (16. * self.ht)
        self.arak_fac = 0.5 / (12. * self.hx * self.hv)
        
        
        # velocity grid
        self.v = v.copy()
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # collision parameter
        self.alpha = coll_freq
        
        # create local vectors
        self.localB   = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        self.localH0  = da1.createLocalVec()
        self.localH1  = da1.createLocalVec()

        # create toolbox object
        self.toolbox = Toolbox(da1, dax, v, nx, nv, ht, hx, hv)
        
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def formMat(self, Mat A, Vec H1):
        cdef np.int64_t i, j, ix
        cdef np.int64_t xe, xs, ye, ys
        
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da1.globalToLocal(H1,      self.localH1)

        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1 = self.da1.getVecArray(self.localH1)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            
            for j in range(0, self.nv):
                
                row.index = (i,)
                row.field = j
                
                if j == 0 or j == self.nv-1:
                    A.setValueStencil(row, row, 1.0)
                
                else:
                    for index, field, value in [
                            ((i-1,), j-1, 1. * self.time_fac - (h[ix-1, j  ] - h[ix,   j-1]) * self.arak_fac),
                            ((i-1,), j  , 2. * self.time_fac - (h[ix,   j+1] - h[ix,   j-1]) * self.arak_fac \
                                                             - (h[ix-1, j+1] - h[ix-1, j-1]) * self.arak_fac),
                            ((i-1,), j+1, 1. * self.time_fac - (h[ix,   j+1] - h[ix-1, j  ]) * self.arak_fac),
                            ((i,  ), j-1, 2. * self.time_fac + (h[ix+1, j  ] - h[ix-1, j  ]) * self.arak_fac \
                                                             + (h[ix+1, j-1] - h[ix-1, j-1]) * self.arak_fac),
                            ((i,  ), j  , 4. * self.time_fac),
                            ((i,  ), j+1, 2. * self.time_fac - (h[ix+1, j  ] - h[ix-1, j  ]) * self.arak_fac \
                                                             - (h[ix+1, j+1] - h[ix-1, j+1]) * self.arak_fac),
                            ((i+1,), j-1, 1. * self.time_fac + (h[ix+1, j  ] - h[ix,   j-1]) * self.arak_fac),
                            ((i+1,), j  , 2. * self.time_fac + (h[ix,   j+1] - h[ix,   j-1]) * self.arak_fac \
                                                             + (h[ix+1, j+1] - h[ix+1, j-1]) * self.arak_fac),
                            ((i+1,), j+1, 1. * self.time_fac + (h[ix,   j+1] - h[ix+1, j  ]) * self.arak_fac),
                        ]:
                        
                        col.index = index
                        col.field = field
                        A.setValueStencil(row, col, value)
                        
                
        A.assemble()
        
        
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def formRHS(self, Vec B, Vec Fh, Vec H1):
        cdef np.int64_t ix, jx
        cdef np.int64_t xs, xe, ys, ye
        
        self.da1.globalToLocal(Fh,      self.localFh)
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da1.globalToLocal(H1,      self.localH1)
        
        cdef np.ndarray[np.float64_t, ndim=2] b  = self.da1.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh = self.da1.getVecArray(self.localFh)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1 = self.da1.getVecArray(self.localH1)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h  = h0 + h1
        
        
        (xs, xe), = self.da1.getRanges()
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(0, self.nv):
                
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[iy, j] = 0.0
                    
                else:
                    b[iy, j] = self.toolbox.time_derivative_J1(fh, ix, j) \
                             - 0.5 * self.toolbox.arakawa_J1(fh, h,  ix, j)

