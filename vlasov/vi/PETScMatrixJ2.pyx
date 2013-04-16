'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport DA, SNES, Mat, Vec

from vlasov.Toolbox import Toolbox


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
        
        # create moment vectors
        self.P   = self.dax.createGlobalVec()
        self.N   = self.dax.createGlobalVec()
        self.U   = self.dax.createGlobalVec()
        self.E   = self.dax.createGlobalVec()
        self.A   = self.dax.createGlobalVec()

        self.Ph  = self.dax.createGlobalVec()
        self.Nh  = self.dax.createGlobalVec()
        self.Uh  = self.dax.createGlobalVec()
        self.Eh  = self.dax.createGlobalVec()
        self.Ah  = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1  = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localH2  = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()
        self.localF   = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()

        self.localP   = dax.createLocalVec()
        self.localN   = dax.createLocalVec()
        self.localU   = dax.createLocalVec()
        self.localE   = dax.createLocalVec()
        self.localA   = dax.createLocalVec()
        
        self.localPh  = dax.createLocalVec()
        self.localNh  = dax.createLocalVec()
        self.localUh  = dax.createLocalVec()
        self.localEh  = dax.createLocalVec()
        self.localAh  = dax.createLocalVec()
        
        # kinetic Hamiltonian
        H0.copy(self.H0)
        self.H2.set(0.)
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1, Vec P, Vec N, Vec U, Vec E, Vec A):
        H1.copy(self.H1h)
        F.copy(self.Fh)
        P.copy(self.Ph)
        N.copy(self.Nh)
        U.copy(self.Uh)
        E.copy(self.Eh)
        A.copy(self.Ah)
        
    
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

        self.dax.globalToLocal(self.Nh,  self.localNh)
        self.dax.globalToLocal(self.Uh,  self.localUh)
        self.dax.globalToLocal(self.Eh,  self.localEh)
        self.dax.globalToLocal(self.Ah,  self.localAh)
        
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getVecArray(self.localH2h)[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Nh  = self.dax.getVecArray(self.localNh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Uh  = self.dax.getVecArray(self.localUh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Eh  = self.dax.getVecArray(self.localEh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ah  = self.dax.getVecArray(self.localAh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1h + h2h
        
        cdef np.float64_t time_fac = 1.0 / (16. * self.ht)
        cdef np.float64_t arak_fac_J2 = 0.5 / (24. * self.hx * self.hv)
        
        cdef np.float64_t coll1_fac = - 0.5 * self.nu * 0.25 / self.hv
        cdef np.float64_t coll2_fac = - 0.5 * self.nu * 0.25 * self.hv2_inv
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        (xs, xe), = self.da2.getRanges()
        
        
        # Poisson equation
        for i in np.arange(xs, xe):
            row.index = (i,)
            row.field = self.nv
            
            # charge density
            for index, value in [
                    ((i-2,), 0.25 * self.charge),
                    ((i,  ), 0.50 * self.charge),
                    ((i+2,), 0.25 * self.charge),
                ]:
                    
                col.index = index
                col.field = self.nv+1
                A.setValueStencil(row, col, value)
            
            
            # Laplace operator
            for index, value in [
                    ((i-2,), - 0.25 * self.hx2_inv),
                    ((i,  ), + 0.50 * self.hx2_inv),
                    ((i+2,), - 0.25 * self.hx2_inv),
                ]:
                
                col.index = index
                col.field = self.nv
                A.setValueStencil(row, col, value)
                    
            
        # moments
        for i in np.arange(xs, xe):
            ix = i-xs+2
            
            row.index = (i,)
            col.index = (i,)
            
            
            # density
            row.field = self.nv+1
            col.field = self.nv+1
            
            A.setValueStencil(row, col, 1. / self.hv)
            
            for j in np.arange(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - 1.)
             
            
            # average velocity density
            row.field = self.nv+2
            col.field = self.nv+2
            
            A.setValueStencil(row, col, 1. / self.hv)
            
            for j in np.arange(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - self.v[j])
            
            
            # average energy density
            row.field = self.nv+3
            col.field = self.nv+3
            
            A.setValueStencil(row, col, 1. / self.hv)
            
            for j in np.arange(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - self.v[j]**2)
                
            
            # temperature
            row.field = self.nv+4
            col.field = self.nv+4
            
            A.setValueStencil(row, col, 1.)
        
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            
            row.index = (i,)
                
            # Vlasov equation
            for j in np.arange(0, self.nv):
                row.field = j
                
                # Dirichlet boundary conditions
                if j <= 1 or j >= self.nv-2:
                    A.setValueStencil(row, row, 1.0)
                    
                else:
                    
                    for index, field, value in [
                            ((i-2,), j  , 1. * time_fac - (h[ix-1, j+1] - h[ix-1, j-1]) * arak_fac_J2),
                            ((i-1,), j-1, 2. * time_fac - (h[ix-2, j  ] - h[ix,   j-2]) * arak_fac_J2 \
                                                        - (h[ix-1, j+1] - h[ix+1, j-1]) * arak_fac_J2 \
                                                        - 1. * coll1_fac * ( Nh[ix-1] * v[j-1] - Uh[ix-1] ) * Ah[ix-1] \
                                                        + 1. * coll2_fac),
                            ((i-1,), j  ,               - 2. * coll2_fac),
                            ((i-1,), j+1, 2. * time_fac - (h[ix,   j+2] - h[ix-2, j  ]) * arak_fac_J2 \
                                                        - (h[ix+1, j+1] - h[ix-1, j-1]) * arak_fac_J2 \
                                                        + 1. * coll1_fac * ( Nh[ix-1] * v[j+1] - Uh[ix-1] ) * Ah[ix-1] \
                                                        + 1. * coll2_fac),
                            ((i,  ), j-2, 1. * time_fac + (h[ix+1, j-1] - h[ix-1, j-1]) * arak_fac_J2),
                            ((i,  ), j-1,               - 2. * coll1_fac * ( Nh[ix  ] * v[j-1] - Uh[ix  ] ) * Ah[ix  ] \
                                                        + 2. * coll2_fac),
                            ((i,  ), j  , 4. * time_fac - 4. * coll2_fac),
                            ((i,  ), j+1,               + 2. * coll1_fac * ( Nh[ix  ] * v[j+1] - Uh[ix  ] ) * Ah[ix  ] \
                                                        + 2. * coll2_fac),
                            ((i,  ), j+2, 1. * time_fac - (h[ix+1, j+1] - h[ix-1, j+1]) * arak_fac_J2),
                            ((i+1,), j-1, 2. * time_fac + (h[ix+2, j  ] - h[ix,   j-2]) * arak_fac_J2 \
                                                        + (h[ix+1, j+1] - h[ix-1, j-1]) * arak_fac_J2 \
                                                        - 1. * coll1_fac * ( Nh[ix+1] * v[j-1] - Uh[ix+1] ) * Ah[ix+1] \
                                                        + 1. * coll2_fac),
                            ((i+1,), j  ,               - 2. * coll2_fac),
                            ((i+1,), j+1, 2. * time_fac + (h[ix,   j+2] - h[ix+2, j  ]) * arak_fac_J2 \
                                                        + (h[ix-1, j+1] - h[ix+1, j-1]) * arak_fac_J2 \
                                                        + 1. * coll1_fac * ( Nh[ix+1] * v[j+1] - Uh[ix+1] ) * Ah[ix+1] \
                                                        + 1. * coll2_fac),
                            ((i+2,), j,   1. * time_fac + (h[ix+1, j+1] - h[ix+1, j-1]) * arak_fac_J2),
                            
                            ((i-2,), self.nv,    + 1. * (fh[ix-1, j+1] - fh[ix-1, j-1]) * arak_fac_J2),
                            ((i-1,), self.nv,    + 1. * (fh[ix-1, j+1] - fh[ix+1, j-1]) * arak_fac_J2 \
                                                 + 1. * (fh[ix+1, j+1] - fh[ix-1, j-1]) * arak_fac_J2 \
                                                 + 1. * (fh[ix-2, j  ] - fh[ix,   j-2]) * arak_fac_J2 \
                                                 + 1. * (fh[ix,   j+2] - fh[ix-2, j  ]) * arak_fac_J2),
                            ((i,  ), self.nv,    + 1. * (fh[ix+1, j+1] - fh[ix-1, j+1]) * arak_fac_J2 \
                                                 - 1. * (fh[ix+1, j-1] - fh[ix-1, j-1]) * arak_fac_J2),
                            ((i+1,), self.nv,    - 1. * (fh[ix-1, j+1] - fh[ix+1, j-1]) * arak_fac_J2 \
                                                 - 1. * (fh[ix+1, j+1] - fh[ix-1, j-1]) * arak_fac_J2 \
                                                 - 1. * (fh[ix,   j+2] - fh[ix+2, j  ]) * arak_fac_J2 \
                                                 - 1. * (fh[ix+2, j  ] - fh[ix,   j-2]) * arak_fac_J2),
                            ((i+2,), self.nv,    - 1. * (fh[ix+1, j+1] - fh[ix+1, j-1]) * arak_fac_J2),
                        ]:
                        
                        col.index = index
                        col.field = field
                        A.setValueStencil(row, col, value)

                
        A.assemble()
                    
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
        
        
        
    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.int64_t i, j, ix, iy, xs, xe
        
        B.set(0.)
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da1.globalToLocal(self.H2, self.localH2)
        self.da1.globalToLocal(self.Fh, self.localFh)
        
        self.dax.globalToLocal(self.Nh,  self.localNh)
        self.dax.globalToLocal(self.Uh,  self.localUh)
        self.dax.globalToLocal(self.Eh,  self.localEh)
        self.dax.globalToLocal(self.Ah,  self.localAh)
        
        cdef np.ndarray[np.float64_t, ndim=2] b  = self.da2.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2 = self.da1.getVecArray(self.localH2)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh = self.da1.getVecArray(self.localFh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Nh  = self.dax.getVecArray(self.localNh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Uh  = self.dax.getVecArray(self.localUh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Eh  = self.dax.getVecArray(self.localEh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ah  = self.dax.getVecArray(self.localAh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h2
        
        cdef np.float64_t nmean = self.Nh.sum() / self.nx
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Poisson equation
            b[iy, self.nv] = nmean * self.charge
            
            
            # moments
            b[iy, self.nv+1] = 0.
            b[iy, self.nv+2] = 0.
            b[iy, self.nv+3] = 0.
            b[iy, self.nv+4] = 0.
#             b[iy, self.nv+4] = Nh[ix] / (Nh[ix] * Eh[ix] - Uh[ix] * Uh[ix])
            
            
            # Vlasov equation
            for j in np.arange(0, self.nv):
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[iy, j] = 0.0
                    
                else:
                    b[iy, j] = self.toolbox.time_derivative_J2(fh, ix, j) \
                             - 0.5 * self.toolbox.arakawa_J2(fh, h, ix, j) \
                             + 0.5 * self.nu * self.toolbox.collT2(fh, ix, j)




    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.mult(X, Y)
        
    
    def mult(self, Vec X, Vec Y):
        (xs, xe), = self.da2.getRanges()
        
        x  = self.da2.getVecArray(X)
        h1 = self.da1.getVecArray(self.H1)
        f  = self.da1.getVecArray(self.F)
        p  = self.dax.getVecArray(self.P)
        n  = self.dax.getVecArray(self.N)
        u  = self.dax.getVecArray(self.U)
        e  = self.dax.getVecArray(self.E)
        a  = self.dax.getVecArray(self.A)
        
        f[xs:xe] = x[xs:xe, 0:self.nv  ]
        p[xs:xe] = x[xs:xe,   self.nv  ]
        n[xs:xe] = x[xs:xe,   self.nv+1]
        u[xs:xe] = x[xs:xe,   self.nv+2]
        e[xs:xe] = x[xs:xe,   self.nv+3]
        a[xs:xe] = x[xs:xe,   self.nv+4]
        
        for j in np.arange(0, self.nv):
            h1[xs:xe, j] = p[xs:xe]
        
        
        self.matrix_mult(Y)
        
        
    @cython.boundscheck(False)
    def matrix_mult(self, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy
        cdef np.uint64_t xe, xs
        
        cdef np.float64_t laplace, integral
        
        cdef np.float64_t nmean = self.N.sum() / self.nx
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.H1,  self.localH1)
        self.da1.globalToLocal(self.H2,  self.localH2)
        self.da1.globalToLocal(self.H1h, self.localH1h)
        self.da1.globalToLocal(self.H2h, self.localH2h)
        self.da1.globalToLocal(self.F,   self.localF )
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.dax.globalToLocal(self.P,   self.localP )
        
        self.dax.globalToLocal(self.N,   self.localN )
        self.dax.globalToLocal(self.U,   self.localU )
        self.dax.globalToLocal(self.E,   self.localE )
        self.dax.globalToLocal(self.A,   self.localA )
        
        self.dax.globalToLocal(self.Nh,  self.localNh)
        self.dax.globalToLocal(self.Uh,  self.localUh)
        self.dax.globalToLocal(self.Eh,  self.localEh)
        self.dax.globalToLocal(self.Ah,  self.localAh)
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1  = self.da1.getVecArray(self.localH1 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2  = self.da1.getVecArray(self.localH2 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getVecArray(self.localH2h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] f   = self.da1.getVecArray(self.localF  )[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        cdef np.ndarray[np.float64_t, ndim=1] p   = self.dax.getVecArray(self.localP  )[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] N   = self.dax.getVecArray(self.localN  )[...]
        cdef np.ndarray[np.float64_t, ndim=1] U   = self.dax.getVecArray(self.localU  )[...]
        cdef np.ndarray[np.float64_t, ndim=1] E   = self.dax.getVecArray(self.localE  )[...]
        cdef np.ndarray[np.float64_t, ndim=1] A   = self.dax.getVecArray(self.localA  )[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Nh  = self.dax.getVecArray(self.localNh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Uh  = self.dax.getVecArray(self.localUh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Eh  = self.dax.getVecArray(self.localEh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ah  = self.dax.getVecArray(self.localAh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (f  + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h     = h0 + h1  + h2
        cdef np.ndarray[np.float64_t, ndim=2] hh    = h0 + h1h + h2h
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Poisson equation
            laplace  = (p[ix-1] + p[ix+1] - 2. * p[ix]) * self.hx2_inv
            integral = 0.25 * ( N[ix-1] + 2. * N[ix] + N[ix+1] )
            
            y[iy, self.nv] = - laplace + self.charge * (integral - nmean)
            
            
            # moments
            y[iy, self.nv+1] = N[ix] / self.hv - (f[ix]            ).sum()
            y[iy, self.nv+2] = U[ix] / self.hv - (f[ix] * self.v   ).sum()
            y[iy, self.nv+3] = E[ix] / self.hv - (f[ix] * self.v**2).sum()
            y[iy, self.nv+4] = A[ix] - Nh[ix] / (Nh[ix] * Eh[ix] - Uh[ix] * Uh[ix])
            
            
            # Vlasov Equation
            for j in np.arange(0, self.nv):
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = f[ix, j]
                    
                else:
                    y[iy, j] = self.toolbox.time_derivative_J2(f,  ix, j) \
                             - self.toolbox.time_derivative_J2(fh, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J2(f, hh, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J2(fh, h, ix, j) \
                             - self.nu * self.toolbox.collT1(f, Nh, Uh, Eh, Ah, ix, j) \
                             - self.nu * self.toolbox.collT2(f_ave, ix, j)


