'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np


from petsc4py import PETSc

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.Toolbox import Toolbox


cdef class PETScMatrix(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, VIDA da1, VIDA da2, VIDA dax, Vec H0,
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
        self.H1p = self.da1.createGlobalVec()
        self.H1h = self.da1.createGlobalVec()
        self.H2p = self.da1.createGlobalVec()
        self.H2h = self.da1.createGlobalVec()
        self.Fp  = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        
        # create moment vectors
        self.Pp  = self.dax.createGlobalVec()
        self.Np  = self.dax.createGlobalVec()
        self.Up  = self.dax.createGlobalVec()
        self.Ep  = self.dax.createGlobalVec()
        self.Ap  = self.dax.createGlobalVec()

        self.Ph  = self.dax.createGlobalVec()
        self.Nh  = self.dax.createGlobalVec()
        self.Uh  = self.dax.createGlobalVec()
        self.Eh  = self.dax.createGlobalVec()
        self.Ah  = self.dax.createGlobalVec()
        
        self.Nc  = self.dax.createGlobalVec()
        self.Uc  = self.dax.createGlobalVec()
        self.Ec  = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1p = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localH2p = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()
        self.localFp  = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()

        self.localPp  = dax.createLocalVec()
        self.localNp  = dax.createLocalVec()
        self.localUp  = dax.createLocalVec()
        self.localEp  = dax.createLocalVec()
        self.localAp  = dax.createLocalVec()
        
        self.localPh  = dax.createLocalVec()
        self.localNh  = dax.createLocalVec()
        self.localUh  = dax.createLocalVec()
        self.localEh  = dax.createLocalVec()
        self.localAh  = dax.createLocalVec()
        
        self.localNc = dax.createLocalVec()
        self.localUc = dax.createLocalVec()
        self.localEc = dax.createLocalVec()
        
        # kinetic Hamiltonian
        H0.copy(self.H0)
        self.H2p.set(0.)
        
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
        self.H2p.copy(self.H2h)
        self.toolbox.potential_to_hamiltonian(Pext, self.H2p)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j, ix
        cdef np.int64_t xe, xs
        
        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
        
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getLocalArray(self.Fh,  self.localFh)
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getLocalArray(self.H0,  self.localH0)
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
        
        cdef np.ndarray[np.float64_t, ndim=1] Nh  = self.dax.getLocalArray(self.Nh,  self.localNh)
        cdef np.ndarray[np.float64_t, ndim=1] Uh  = self.dax.getLocalArray(self.Uh,  self.localUh)
        cdef np.ndarray[np.float64_t, ndim=1] Eh  = self.dax.getLocalArray(self.Eh,  self.localEh)
        cdef np.ndarray[np.float64_t, ndim=1] Ah  = self.dax.getLocalArray(self.Ah,  self.localAh)
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1h + h2h
        
        cdef np.float64_t time_fac    = 1.0  / self.ht
#         cdef np.float64_t arak_fac_J1 = 0.25 / (12. * self.hx * self.hv)
#         cdef np.float64_t arak_fac_J2 = 0.25 / (24. * self.hx * self.hv)
        cdef np.float64_t arak_fac_J1 = + 1.0 / (12. * self.hx * self.hv)
        cdef np.float64_t arak_fac_J2 = - 0.5 / (24. * self.hx * self.hv)
        cdef np.float64_t poisson_fac = self.charge / 36. 
        
        cdef np.float64_t coll1_fac = - 0.5 * self.nu * 0.25 * 0.5 / self.hv
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
#             for index, value in [
#                     ((i-2,), - 0.25 * self.charge),
#                     ((i-1,), + 0.50 * self.charge),
#                     ((i,  ), + 0.50 * self.charge),
#                     ((i+1,), + 0.50 * self.charge),
#                     ((i+2,), - 0.25 * self.charge),
#                 ]:
#             for index, value in [
#                     ((i-2,), 1.  * poisson_fac),
#                     ((i-1,), 8.  * poisson_fac),
#                     ((i,  ), 18. * poisson_fac),
#                     ((i+1,), 8.  * poisson_fac),
#                     ((i+2,), 1.  * poisson_fac),
#                 ]:
#                     
#                 col.index = index
#                 col.field = self.nv+1
#                 A.setValueStencil(row, col, value)
            
            col.index = (i,  )
            col.field = self.nv+1
            A.setValueStencil(row, col, self.charge)
            
            
            # Laplace operator
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
            
            A.setValueStencil(row, col, 1.)
            
            for j in np.arange(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - 1. * self.hv)
             
            
            # average velocity density
            row.field = self.nv+2
            col.field = self.nv+2
            
            A.setValueStencil(row, col, 1.)
            
            for j in np.arange(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - self.v[j] * self.hv)
            
            
            # average energy density
            row.field = self.nv+3
            col.field = self.nv+3
            
            A.setValueStencil(row, col, 1.)
            
            for j in np.arange(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - self.v[j]**2 * self.hv)
                
            
        
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
                            ((i-2,), j  , - (h[ix-1, j+1] - h[ix-1, j-1]) * arak_fac_J2),
                            ((i-1,), j-1, - (h[ix-1, j  ] - h[ix,   j-1]) * arak_fac_J1 \
                                          - (h[ix-2, j  ] - h[ix,   j-2]) * arak_fac_J2 \
                                          - (h[ix-1, j+1] - h[ix+1, j-1]) * arak_fac_J2),
                            ((i-1,), j  , - (h[ix,   j+1] - h[ix,   j-1]) * arak_fac_J1 \
                                          - (h[ix-1, j+1] - h[ix-1, j-1]) * arak_fac_J1),
                            ((i-1,), j+1, - (h[ix,   j+1] - h[ix-1, j  ]) * arak_fac_J1 \
                                          - (h[ix,   j+2] - h[ix-2, j  ]) * arak_fac_J2 \
                                          - (h[ix+1, j+1] - h[ix-1, j-1]) * arak_fac_J2),
                            ((i,  ), j-2, + (h[ix+1, j-1] - h[ix-1, j-1]) * arak_fac_J2),
                            ((i,  ), j-1, + (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac_J1 \
                                          + (h[ix+1, j-1] - h[ix-1, j-1]) * arak_fac_J1 \
                                          - 1. * coll1_fac * ( Nh[ix  ] * v[j-1] - Uh[ix  ] ) * Ah[ix  ] \
                                          + 1. * coll2_fac),
                            ((i,  ), j  , + time_fac \
                                          - 2. * coll2_fac),
                            ((i,  ), j+1, - (h[ix+1, j  ] - h[ix-1, j  ]) * arak_fac_J1 \
                                          - (h[ix+1, j+1] - h[ix-1, j+1]) * arak_fac_J1 \
                                          + 1. * coll1_fac * ( Nh[ix  ] * v[j+1] - Uh[ix  ] ) * Ah[ix  ] \
                                          + 1. * coll2_fac),
                            ((i,  ), j+2, - (h[ix+1, j+1] - h[ix-1, j+1]) * arak_fac_J2),
                            ((i+1,), j-1, + (h[ix+1, j  ] - h[ix,   j-1]) * arak_fac_J1 \
                                          + (h[ix+2, j  ] - h[ix,   j-2]) * arak_fac_J2 \
                                          + (h[ix+1, j+1] - h[ix-1, j-1]) * arak_fac_J2),
                            ((i+1,), j  , + (h[ix,   j+1] - h[ix,   j-1]) * arak_fac_J1 \
                                          + (h[ix+1, j+1] - h[ix+1, j-1]) * arak_fac_J1),
                            ((i+1,), j+1, + (h[ix,   j+1] - h[ix+1, j  ]) * arak_fac_J1 \
                                          + (h[ix,   j+2] - h[ix+2, j  ]) * arak_fac_J2 \
                                          + (h[ix-1, j+1] - h[ix+1, j-1]) * arak_fac_J2),
                            ((i+2,), j  , + (h[ix+1, j+1] - h[ix+1, j-1]) * arak_fac_J2),
                            
                            ((i-2,), self.nv,    + 1. * (fh[ix-1, j+1] - fh[ix-1, j-1]) * arak_fac_J2),
                            ((i-1,), self.nv,    + 2. * (fh[ix,   j+1] - fh[ix,   j-1]) * arak_fac_J1 \
                                                 + 1. * (fh[ix-1, j+1] - fh[ix-1, j-1]) * arak_fac_J1 \
                                                 + 1. * (fh[ix-1, j+1] - fh[ix+1, j-1]) * arak_fac_J2 \
                                                 + 1. * (fh[ix+1, j+1] - fh[ix-1, j-1]) * arak_fac_J2 \
                                                 + 1. * (fh[ix-2, j  ] - fh[ix,   j-2]) * arak_fac_J2 \
                                                 + 1. * (fh[ix,   j+2] - fh[ix-2, j  ]) * arak_fac_J2),
                            ((i,  ), self.nv,    - 1. * (fh[ix+1, j-1] - fh[ix-1, j-1]) * arak_fac_J1 \
                                                 + 1. * (fh[ix+1, j+1] - fh[ix-1, j+1]) * arak_fac_J1 \
                                                 + 1. * (fh[ix+1, j+1] - fh[ix-1, j+1]) * arak_fac_J2 \
                                                 - 1. * (fh[ix+1, j-1] - fh[ix-1, j-1]) * arak_fac_J2),
                            ((i+1,), self.nv,    - 2. * (fh[ix,   j+1] - fh[ix,   j-1]) * arak_fac_J1 \
                                                 - 1. * (fh[ix+1, j+1] - fh[ix+1, j-1]) * arak_fac_J1 \
                                                 - 1. * (fh[ix-1, j+1] - fh[ix+1, j-1]) * arak_fac_J2 \
                                                 - 1. * (fh[ix+1, j+1] - fh[ix-1, j-1]) * arak_fac_J2 \
                                                 - 1. * (fh[ix,   j+2] - fh[ix+2, j  ]) * arak_fac_J2 \
                                                 - 1. * (fh[ix+2, j  ] - fh[ix,   j-2]) * arak_fac_J2),
                            ((i+2,), self.nv,    - 1. * (fh[ix+1, j+1] - fh[ix+1, j-1]) * arak_fac_J2),
                        ]:
                        
                        col.index = index
                        col.field = field
                        A.setValueStencil(row, col, value)

                
        A.assemble()
                    
        
        
    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.int64_t i, j, ix, iy, xs, xe
        
        B.set(0.)
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.H2p, self.localH2p)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        
        self.dax.globalToLocal(self.Nh,  self.localNh)
        self.dax.globalToLocal(self.Uh,  self.localUh)
        self.dax.globalToLocal(self.Eh,  self.localEh)
        self.dax.globalToLocal(self.Ah,  self.localAh)
        
        cdef np.ndarray[np.float64_t, ndim=2] b   = self.da2.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2p = self.da1.getVecArray(self.localH2p)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Nh  = self.dax.getVecArray(self.localNh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Uh  = self.dax.getVecArray(self.localUh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Eh  = self.dax.getVecArray(self.localEh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ah  = self.dax.getVecArray(self.localAh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h2p
        
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
            
            
            # Vlasov equation
            for j in np.arange(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    b[iy, j] = 0.0
                    
                else:
                    b[iy, j] = self.toolbox.time_derivative_woa(fh, ix, j) \
                             - 0.5 * self.toolbox.arakawa_J4(fh, h, ix, j) \
                             + 0.5 * self.nu * self.toolbox.collT1woa(fh, Nh, Uh, Eh, Ah, ix, j) \
                             + 0.5 * self.nu * self.toolbox.collT2woa(fh, Nh, Uh, Eh, Ah, ix, j)




    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.mult(X, Y)
        
    
    def mult(self, Vec X, Vec Y):
        (xs, xe), = self.da2.getRanges()
        
        x  = self.da2.getVecArray(X)
        h1 = self.da1.getVecArray(self.H1p)
        f  = self.da1.getVecArray(self.Fp)
        p  = self.dax.getVecArray(self.Pp)
        n  = self.dax.getVecArray(self.Np)
        u  = self.dax.getVecArray(self.Up)
        e  = self.dax.getVecArray(self.Ep)
        a  = self.dax.getVecArray(self.Ap)
        
        f[xs:xe] = x[xs:xe, 0:self.nv  ]
        p[xs:xe] = x[xs:xe,   self.nv  ]
        n[xs:xe] = x[xs:xe,   self.nv+1]
        u[xs:xe] = x[xs:xe,   self.nv+2]
        e[xs:xe] = x[xs:xe,   self.nv+3]
        
        for i in range(xs,xe):
            a[i] = n[i] / ( n[i] * e[i] - u[i]**2)
        
        for j in np.arange(0, self.nv):
            h1[xs:xe, j] = p[xs:xe]
        
        
        self.matrix_mult(Y)
        
        
    @cython.boundscheck(False)
    def matrix_mult(self, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy
        cdef np.uint64_t xe, xs
        
        cdef np.float64_t nmean = self.Np.sum() / self.nx
        
        self.toolbox.compute_density(self.Fp, self.Nc)
        self.toolbox.compute_velocity_density(self.Fp, self.Uc)
        self.toolbox.compute_energy_density(self.Fp, self.Ec)
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.H1p, self.localH1p)
        self.da1.globalToLocal(self.H2p, self.localH2p)
        self.da1.globalToLocal(self.H1h, self.localH1h)
        self.da1.globalToLocal(self.H2h, self.localH2h)
        self.da1.globalToLocal(self.Fp,  self.localFp)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.dax.globalToLocal(self.Pp,  self.localPp)
        
        self.dax.globalToLocal(self.Np,  self.localNp)
        self.dax.globalToLocal(self.Up,  self.localUp)
        self.dax.globalToLocal(self.Ep,  self.localEp)
        self.dax.globalToLocal(self.Ap,  self.localAp)
        
        self.dax.globalToLocal(self.Nh,  self.localNh)
        self.dax.globalToLocal(self.Uh,  self.localUh)
        self.dax.globalToLocal(self.Eh,  self.localEh)
        self.dax.globalToLocal(self.Ah,  self.localAh)
        
        self.dax.globalToLocal(self.Nc,  self.localNc)
        self.dax.globalToLocal(self.Uc,  self.localUc)
        self.dax.globalToLocal(self.Ec,  self.localEc)
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1p = self.da1.getVecArray(self.localH1p)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2p = self.da1.getVecArray(self.localH2p)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getVecArray(self.localH2h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fp  = self.da1.getVecArray(self.localFp )[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        cdef np.ndarray[np.float64_t, ndim=1] pp  = self.dax.getVecArray(self.localPp )[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Np  = self.dax.getVecArray(self.localNp )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Up  = self.dax.getVecArray(self.localUp )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ep  = self.dax.getVecArray(self.localEp )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ap  = self.dax.getVecArray(self.localAp )[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Nh  = self.dax.getVecArray(self.localNh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Uh  = self.dax.getVecArray(self.localUh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Eh  = self.dax.getVecArray(self.localEh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ah  = self.dax.getVecArray(self.localAh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Nc = self.dax.getVecArray(self.localNc)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Uc = self.dax.getVecArray(self.localUc)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ec = self.dax.getVecArray(self.localEc)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef np.ndarray[np.float64_t, ndim=2] hp    = h0 + h1p + h2p
        cdef np.ndarray[np.float64_t, ndim=2] hh    = h0 + h1h + h2h
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Poisson equation
            y[iy, self.nv] = - ( pp[ix-1] + pp[ix+1] - 2. * pp[ix] ) * self.hx2_inv + self.charge * (Np[ix] - nmean)
            
            
            # moments
            y[iy, self.nv+1] = Np[ix] - Nc[ix]
            y[iy, self.nv+2] = Up[ix] - Uc[ix]
            y[iy, self.nv+3] = Ep[ix] - Ec[ix]
            
            
            # Vlasov Equation
            for j in np.arange(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fp[ix, j]
                    
                else:
                    y[iy, j] = self.toolbox.time_derivative_woa(fp, ix, j) \
                             - self.toolbox.time_derivative_woa(fh, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(fp, hh, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(fh, hp, ix, j) \
                             - 0.5 * self.nu * self.toolbox.collT1woa(fp, Nh, Uh, Eh, Ah, ix, j) \
                             - 0.5 * self.nu * self.toolbox.collT1woa(fh, Nh, Uh, Eh, Ah, ix, j) \
                             - 0.5 * self.nu * self.toolbox.collT2woa(fp, Nh, Uh, Eh, Ah, ix, j) \
                             - 0.5 * self.nu * self.toolbox.collT2woa(fh, Nh, Uh, Eh, Ah, ix, j)
