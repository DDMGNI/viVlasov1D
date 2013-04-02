'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.vi.Toolbox import Toolbox


cdef class PETScJacobian(object):
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
        
        # poisson constant
        self.charge = charge
        
        # collision frequency
        self.nu = coll_freq
        
        # velocity grid
        self.v = v.copy()
        
        # create work and history vectors
        self.H0  = self.da1.createGlobalVec()
        self.H1p = self.da1.createGlobalVec()
        self.H1h = self.da1.createGlobalVec()
        self.H2  = self.da1.createGlobalVec()
        self.H2h = self.da1.createGlobalVec()
        self.Fp  = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        self.Pp = self.dax.createGlobalVec()
        
        # create moment vectors
        self.Np = self.dax.createGlobalVec()
        self.Up = self.dax.createGlobalVec()
        self.Ep = self.dax.createGlobalVec()
        self.Ap = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1p = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localH2  = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()
        self.localFp  = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()

        self.localNp = dax.createLocalVec()
        self.localUp = dax.createLocalVec()
        self.localEp = dax.createLocalVec()
        self.localAp = dax.createLocalVec()
        
        # kinetic Hamiltonian
        H0.copy(self.H0)
        
        # external Hamiltonian
        self.H2.set(0.)
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
        
    
    def update_history(self, Vec F, Vec H1):
        F.copy(self.Fh)
        H1.copy(self.H1h)
        
    
    def update_previous(self, Vec X):
        (xs, xe), = self.da2.getRanges()
        
        x  = self.da2.getVecArray(X)
        h1 = self.da1.getVecArray(self.H1p)
        f  = self.da1.getVecArray(self.Fp)
        p  = self.dax.getVecArray(self.Pp)
        n  = self.dax.getVecArray(self.Np)
        u  = self.dax.getVecArray(self.Up)
        e  = self.dax.getVecArray(self.Ep)
        a  = self.dax.getVecArray(self.Ap)
        
        f[xs:xe] = x[xs:xe, 0:self.nv]
        p[xs:xe] = x[xs:xe,   self.nv]
        n[xs:xe] = x[xs:xe,   self.nv+1]
        u[xs:xe] = x[xs:xe,   self.nv+2]
        e[xs:xe] = x[xs:xe,   self.nv+3]
        a[xs:xe] = x[xs:xe,   self.nv+4]
        
        for j in np.arange(0, self.nv):
            h1[xs:xe, j] = p[xs:xe]
        
        
    
    def update_external(self, Vec Pext):
        self.H2.copy(self.H2h)
        self.toolbox.potential_to_hamiltonian(Pext, self.H2)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j, ix
        cdef np.int64_t xe, xs
        
        cdef np.float64_t afac
        
        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(self.Fp,  self.localFp)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.H1p, self.localH1p)
        self.da1.globalToLocal(self.H1h, self.localH1h)
        self.da1.globalToLocal(self.H2,  self.localH2 )
        self.da1.globalToLocal(self.H2h, self.localH2h)
        
        self.dax.globalToLocal(self.Np,  self.localNp)
        self.dax.globalToLocal(self.Up,  self.localUp)
        self.dax.globalToLocal(self.Ep,  self.localEp)
        self.dax.globalToLocal(self.Ap,  self.localAp)
        
        cdef np.ndarray[np.float64_t, ndim=2] fp  = self.da1.getVecArray(self.localFp) [...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h1p = self.da1.getVecArray(self.localH1p)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2  = self.da1.getVecArray(self.localH2 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getVecArray(self.localH2h)[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Np = self.dax.getVecArray(self.localNp)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Up = self.dax.getVecArray(self.localUp)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ep = self.dax.getVecArray(self.localEp)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ap = self.dax.getVecArray(self.localAp)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h_ave = h0 + 0.5 * (h1p + h1h) + 0.5 * (h2 + h2h)
        
        
#        cdef np.float64_t time_fac = 0.
#        cdef np.float64_t arak_fac = 0.
#        cdef np.float64_t poss_fac = 0.
#        cdef np.float64_t coll1_fac = 0.
#        cdef np.float64_t coll2_fac = 0.
        
        cdef np.float64_t time_fac = 1.0 / (16. * self.ht)
        cdef np.float64_t arak_fac = 0.5 / (12. * self.hx * self.hv)
        cdef np.float64_t poss_fac = 0.25 * self.charge
        
        cdef np.float64_t coll1_fac = - 0.5 * self.nu * 0.25 * 0.5 / self.hv
        cdef np.float64_t coll2_fac = - 0.5 * self.nu * 0.25 * self.hv2_inv
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        # Poisson equation
        for i in np.arange(xs, xe):
            row.index = (i,)
            row.field = self.nv
            
            if i == 0:
                col.index = (i,)
                col.field = self.nv
                A.setValueStencil(row, col, 1.)
                
            else:
                # charge density
                for index, value in [
                        ((i-1,), 1. * poss_fac),
                        ((i,  ), 2. * poss_fac),
                        ((i+1,), 1. * poss_fac),
                    ]:
                    
                    col.index = index
                    col.field = self.nv+1
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
                    
        
        
        # moments
        for i in np.arange(xs, xe):
            ix = i-xs+1
            
            row.index = (i,)
            col.index = (i,)
            
            # density
            row.field = self.nv+1
            
            A.setValueStencil(row, row, 1.)
            
            for j in np.arange(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - self.hv)
            
            
            # average velocity density
            row.field = self.nv+2
            
            A.setValueStencil(row, row, 1.)
            
            for j in np.arange(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - self.v[j] * self.hv)
            
                
            # average energy density
            row.field = self.nv+3
            
            A.setValueStencil(row, row, 1.)
            
            for j in np.arange(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - self.v[j] * self.v[j] * self.hv)
            
                
            # temperature
            row.field = self.nv+4
            
            afac = ( Np[ix] * Ep[ix] - Up[ix] * Up[ix] )
            
            A.setValueStencil(row, row, 1.)
            
            col.field = self.nv+1
            A.setValueStencil(row, col, - 1. / afac - Np[ix] * Ep[ix] / afac**2)
            
            col.field = self.nv+2
            A.setValueStencil(row, col, + 2. * Np[ix] * Up[ix] / afac**2)
            
            col.field = self.nv+3
            A.setValueStencil(row, col, - 1. * Np[ix] * Np[ix] * Ep[ix] / afac**2)
        
        
        
        # Vlasov Equation
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
                            ((i-1,), j-1, 1. * time_fac - (h_ave[ix-1, j  ] - h_ave[ix,   j-1]) * arak_fac \
                                                        - 1. * coll1_fac * ( Np[ix-1] * v[j-1] - Up[ix-1] ) * Ap[ix-1] \
                                                        + 1. * coll2_fac),
                            ((i-1,), j  , 2. * time_fac - (h_ave[ix,   j+1] - h_ave[ix,   j-1]) * arak_fac \
                                                        - (h_ave[ix-1, j+1] - h_ave[ix-1, j-1]) * arak_fac \
                                                        - 2. * coll2_fac),
                            ((i-1,), j+1, 1. * time_fac - (h_ave[ix,   j+1] - h_ave[ix-1, j  ]) * arak_fac \
                                                        + 1. * coll1_fac * ( Np[ix-1] * v[j+1] - Up[ix-1] ) * Ap[ix-1] \
                                                        + 1. * coll2_fac),
                            ((i,  ), j-1, 2. * time_fac + (h_ave[ix+1, j  ] - h_ave[ix-1, j  ]) * arak_fac \
                                                        + (h_ave[ix+1, j-1] - h_ave[ix-1, j-1]) * arak_fac \
                                                        - 2. * coll1_fac * ( Np[ix  ] * v[j-1] - Up[ix  ] ) * Ap[ix  ] \
                                                        + 2. * coll2_fac),
                            ((i,  ), j  , 4. * time_fac \
                                                        - 4. * coll2_fac),
                            ((i,  ), j+1, 2. * time_fac - (h_ave[ix+1, j  ] - h_ave[ix-1, j  ]) * arak_fac \
                                                        - (h_ave[ix+1, j+1] - h_ave[ix-1, j+1]) * arak_fac \
                                                        + 2. * coll1_fac * ( Np[ix  ] * v[j+1] - Up[ix  ] ) * Ap[ix  ] \
                                                        + 2. * coll2_fac),
                            ((i+1,), j-1, 1. * time_fac + (h_ave[ix+1, j  ] - h_ave[ix,   j-1]) * arak_fac \
                                                        - 1. * coll1_fac * ( Np[ix+1] * v[j-1] - Up[ix+1] ) * Ap[ix+1] \
                                                        + 1. * coll2_fac),
                            ((i+1,), j  , 2. * time_fac + (h_ave[ix,   j+1] - h_ave[ix,   j-1]) * arak_fac \
                                                        + (h_ave[ix+1, j+1] - h_ave[ix+1, j-1]) * arak_fac \
                                                        - 2. * coll2_fac),
                            ((i+1,), j+1, 1. * time_fac + (h_ave[ix,   j+1] - h_ave[ix+1, j  ]) * arak_fac \
                                                        + 1. * coll1_fac * ( Np[ix+1] * v[j+1] - Up[ix+1] ) * Ap[ix+1] \
                                                        + 1. * coll2_fac),
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
