'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport DA, SNES, Mat, Vec

from vlasov.vi.Toolbox import Toolbox


cdef class PETScMatrix(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DA da1, DA da2, DA dax, Vec H0,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 np.float64_t charge, np.float64_t coll_freq=0.):
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
        self.charge = charge
        
        # collision frequency
        self.nu = coll_freq
        
        # create work and history vectors
        self.H0  = self.da1.createGlobalVec()
        self.H1  = self.da1.createGlobalVec()
        self.H1h = self.da1.createGlobalVec()
        self.H2  = self.da1.createGlobalVec()
        self.H2h = self.da1.createGlobalVec()
        self.F   = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        self.H2.set(0.)
        
        # create moment vectors
        self.A1 = self.dax.createGlobalVec()
        self.A2 = self.dax.createGlobalVec()
        self.A3 = self.dax.createGlobalVec()
        self.N  = self.dax.createGlobalVec()
        self.U  = self.dax.createGlobalVec()
        self.E  = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1  = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localH2  = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()
        self.localF   = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        self.localP   = dax.createLocalVec()

        self.localA1 = dax.createLocalVec()
        self.localA2 = dax.createLocalVec()
        self.localA3 = dax.createLocalVec()
        
        # kinetic Hamiltonian
        H0.copy(self.H0)
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1):
        F.copy(self.Fh)
        H1.copy(self.H1h)
        
    
    def update_external(self, Vec Pext):
        self.H2.copy(self.H2h)
        self.toolbox.potential_to_hamiltonian(Pext, self.H2)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j, ix
        cdef np.int64_t xe, xs
        
        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
        
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.H1h, self.localH1h)
        self.da1.globalToLocal(self.H2h, self.localH2h)

        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getVecArray(self.localH2h)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1h + h2h
        
        cdef np.float64_t time_fac = 1.0 / (16. * self.ht)
        cdef np.float64_t arak_fac = 0.5 / (12. * self.hx * self.hv)
        cdef np.float64_t poss_fac = 0.25 * self.hv * self.charge
        
        cdef np.float64_t coll1_fac = - self.nu * 0.25 * 0.5 / self.hv
        cdef np.float64_t coll2_fac = - self.nu * 0.25 * self.hv2_inv
        
#        cdef np.float64_t coll1_fac = - self.nu * 0.5 / self.hv
#        cdef np.float64_t coll2_fac = - self.nu * self.hv2_inv
        
        
        # calculate moments
        self.toolbox.collT_moments(self.Fh, self.A1, self.A2, self.A3, self.N, self.U, self.E)
        
        self.dax.globalToLocal(self.A1, self.localA1)
        self.dax.globalToLocal(self.A2, self.localA2)
        self.dax.globalToLocal(self.A3, self.localA3)
        
        cdef np.ndarray[np.float64_t, ndim=1] A1 = self.dax.getVecArray(self.localA1)[...]
        cdef np.ndarray[np.float64_t, ndim=1] A2 = self.dax.getVecArray(self.localA2)[...]
        cdef np.ndarray[np.float64_t, ndim=1] A3 = self.dax.getVecArray(self.localA3)[...]
        
        
        
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
#                            ((i-1,), j-1, 1. * time_fac - (h[ix-1, j  ] - h[ix,   j-1]) * arak_fac),
#                            ((i-1,), j  , 2. * time_fac - (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
#                                                        - (h[ix-1, j+1] - h[ix-1, j-1]) * arak_fac),
#                            ((i-1,), j+1, 1. * time_fac - (h[ix,   j+1] - h[ix-1, j  ]) * arak_fac),
#                            ((i,  ), j-1, 2. * time_fac + (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
#                                                        + (h[ix+1, j-1] - h[ix-1, j-1]) * arak_fac \
#                                                        - 1. * coll1_fac * ( A1[ix  ] * v[j-1] - A2[ix  ] ) * A1[ix  ] * A3[ix  ] \
#                                                        + 1. * coll2_fac),
#                            ((i,  ), j  , 4. * time_fac \
#                                                        - 2. * coll2_fac),
#                            ((i,  ), j+1, 2. * time_fac - (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
#                                                        - (h[ix+1, j+1] - h[ix-1, j+1]) * arak_fac \
#                                                        + 1. * coll1_fac * ( A1[ix  ] * v[j+1] - A2[ix  ] ) * A1[ix  ] * A3[ix  ] \
#                                                        + 1. * coll2_fac),
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
                    
                    for index, field, value in [
                            ((i-1,), j-1, 1. * time_fac - (h[ix-1, j  ] - h[ix,   j-1]) * arak_fac \
                                                        - 1. * coll1_fac * ( A1[ix-1] * v[j-1] - A2[ix-1] ) * A3[ix-1] \
                                                        + 1. * coll2_fac),
                            ((i-1,), j  , 2. * time_fac - (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
                                                        - (h[ix-1, j+1] - h[ix-1, j-1]) * arak_fac \
                                                        - 2. * coll2_fac),
                            ((i-1,), j+1, 1. * time_fac - (h[ix,   j+1] - h[ix-1, j  ]) * arak_fac \
                                                        + 1. * coll1_fac * ( A1[ix-1] * v[j+1] - A2[ix-1] ) * A3[ix-1] \
                                                        + 1. * coll2_fac),
                            ((i,  ), j-1, 2. * time_fac + (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
                                                        + (h[ix+1, j-1] - h[ix-1, j-1]) * arak_fac \
                                                        - 2. * coll1_fac * ( A1[ix  ] * v[j-1] - A2[ix  ] ) * A3[ix  ] \
                                                        + 2. * coll2_fac),
                            ((i,  ), j  , 4. * time_fac \
                                                        - 4. * coll2_fac),
                            ((i,  ), j+1, 2. * time_fac - (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac \
                                                        - (h[ix+1, j+1] - h[ix-1, j+1]) * arak_fac \
                                                        + 2. * coll1_fac * ( A1[ix  ] * v[j+1] - A2[ix  ] ) * A3[ix  ] \
                                                        + 2. * coll2_fac),
                            ((i+1,), j-1, 1. * time_fac + (h[ix+1, j  ] - h[ix,   j-1]) * arak_fac \
                                                        - 1. * coll1_fac * ( A1[ix+1] * v[j-1] - A2[ix+1] ) * A3[ix+1] \
                                                        + 1. * coll2_fac),
                            ((i+1,), j  , 2. * time_fac + (h[ix,   j+1] - h[ix,   j-1]) * arak_fac \
                                                        + (h[ix+1, j+1] - h[ix+1, j-1]) * arak_fac \
                                                        - 2. * coll2_fac),
                            ((i+1,), j+1, 1. * time_fac + (h[ix,   j+1] - h[ix+1, j  ]) * arak_fac \
                                                        + 1. * coll1_fac * ( A1[ix+1] * v[j+1] - A2[ix+1] ) * A3[ix+1] \
                                                        + 1. * coll2_fac),
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
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da1.globalToLocal(self.H2, self.localH2)
        self.da1.globalToLocal(self.Fh, self.localFh)
        
        cdef np.ndarray[np.float64_t, ndim=2] b  = self.da2.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2 = self.da1.getVecArray(self.localH2)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh = self.da1.getVecArray(self.localFh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h2
        
        cdef np.float64_t fmean = self.Fh.sum() * self.hv / self.nx
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            if i == 0:
                b[iy, self.nv] = 0.
                
            else:
                b[iy, self.nv] = fmean * self.charge
            
            
            # Vlasov equation
            for j in np.arange(0, self.nv):
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[iy, j] = 0.0
                    
                else:
                    b[iy, j] = self.toolbox.time_derivative(fh, ix, j) \
                             - 0.5 * self.toolbox.arakawa(fh, h, ix, j)




    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.mult(X, Y)
        
    
    def mult(self, Vec X, Vec Y):
        (xs, xe), = self.da2.getRanges()
        
        H = self.da1.createGlobalVec()
        F = self.da1.createGlobalVec()
        P = self.dax.createGlobalVec()
        
        x = self.da2.getVecArray(X)
        h = self.da1.getVecArray(H)
        f = self.da1.getVecArray(F)
        p = self.dax.getVecArray(P)
        
        h0 = self.da1.getVecArray(self.H0)
        
        
        f[xs:xe] = x[xs:xe, 0:self.nv]
        p[xs:xe] = x[xs:xe,   self.nv]
        
        for j in np.arange(0, self.nv):
            h[xs:xe, j] = h0[xs:xe, j] + p[xs:xe]
        
        
        self.matrix_mult(F, H, P, Y)
        
        
    @cython.boundscheck(False)
    def matrix_mult(self, Vec F, Vec H, Vec P, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy
        cdef np.uint64_t xe, xs
        
        cdef np.float64_t laplace, integral, nmean, phisum
        
        nmean  = F.sum() * self.hv / self.nx
        phisum = P.sum()
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(F,        self.localF )
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.da1.globalToLocal(H,        self.localH )
        self.da1.globalToLocal(self.Hh,  self.localHh)
#        self.da1.globalToLocal(self.H2,  self.localH2 )
#        self.da1.globalToLocal(self.H2h, self.localH2h)
        self.dax.globalToLocal(P,        self.localP )
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] f   = self.da1.getVecArray(self.localF  )[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h   = self.da1.getVecArray(self.localH  )[...]
        cdef np.ndarray[np.float64_t, ndim=2] hh  = self.da1.getVecArray(self.localHh )[...]
#        cdef np.ndarray[np.float64_t, ndim=2] h2  = self.da1.getVecArray(self.localH2 )[...]
#        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getVecArray(self.localH2h)[...]
        cdef np.ndarray[np.float64_t, ndim=1] p   = self.dax.getVecArray(self.localP  )[...]
        
        
        # calculate moments
        self.toolbox.collT_moments(self.Fh, self.A1, self.A2, self.A3, self.N, self.U, self.E)
        
        self.dax.globalToLocal(self.A1, self.localA1)
        self.dax.globalToLocal(self.A2, self.localA2)
        self.dax.globalToLocal(self.A3, self.localA3)
        
        cdef np.ndarray[np.float64_t, ndim=1] A1 = self.dax.getVecArray(self.localA1)[...]
        cdef np.ndarray[np.float64_t, ndim=1] A2 = self.dax.getVecArray(self.localA2)[...]
        cdef np.ndarray[np.float64_t, ndim=1] A3 = self.dax.getVecArray(self.localA3)[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            
            if i == 0:
                y[iy, self.nv] = p[ix]
            
            else:
                    
                laplace  = (p[ix-1] + p[ix+1] - 2. * p[ix]) * self.hx2_inv
                
                integral = ( \
                             + 1. * f[ix-1, :].sum() \
                             + 2. * f[ix,   :].sum() \
                             + 1. * f[ix+1, :].sum() \
                           ) * 0.25 * self.hv
                
                y[iy, self.nv] = - laplace + self.charge * (integral - nmean)
            
            # Vlasov Equation
            for j in np.arange(0, self.nv):
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = f[ix,j]
                    
                else:
                    y[iy, j] = self.toolbox.time_derivative(f,  ix, j) \
                             - self.toolbox.time_derivative(fh, ix, j) \
                             + 0.5 * self.toolbox.arakawa(f, hh, ix, j) \
                             + 0.5 * self.toolbox.arakawa(fh, h, ix, j) \
                             - self.nu * self.toolbox.collT1(f, A1, A2, A3, ix, j) \
                             - self.nu * self.toolbox.collT2(f, ix, j)


