'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.Toolbox import Toolbox


cdef class PETScFunction(object):
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
        self.H2p = self.da1.createGlobalVec()
        self.H2h = self.da1.createGlobalVec()
        self.Fp  = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        
        # create spatial vectors
        self.Pp  = self.dax.createGlobalVec()
        self.Np  = self.dax.createGlobalVec()
        self.NUp = self.dax.createGlobalVec()
        self.NEp = self.dax.createGlobalVec()
        self.Up  = self.dax.createGlobalVec()
        self.Ep  = self.dax.createGlobalVec()
        self.Ap  = self.dax.createGlobalVec()

        self.Ph  = self.dax.createGlobalVec()
        self.Nh  = self.dax.createGlobalVec()
        self.NUh = self.dax.createGlobalVec()
        self.NEh = self.dax.createGlobalVec()
        self.Uh  = self.dax.createGlobalVec()
        self.Eh  = self.dax.createGlobalVec()
        self.Ah  = self.dax.createGlobalVec()
        
        self.Nc  = self.dax.createGlobalVec()
        self.NUc = self.dax.createGlobalVec()
        self.NEc = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1p = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localH2p = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()
        self.localFp  = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        
        # create local spatial vectors
        self.localPp  = dax.createLocalVec()
        self.localNp  = dax.createLocalVec()
        self.localNUp = dax.createLocalVec()
        self.localNEp = dax.createLocalVec()
        self.localUp  = dax.createLocalVec()
        self.localEp  = dax.createLocalVec()
        self.localAp  = dax.createLocalVec()
        
        self.localPh  = dax.createLocalVec()
        self.localNh  = dax.createLocalVec()
        self.localNUh = dax.createLocalVec()
        self.localNEh = dax.createLocalVec()
        self.localUh  = dax.createLocalVec()
        self.localEh  = dax.createLocalVec()
        self.localAh  = dax.createLocalVec()
        
        self.localNc  = dax.createLocalVec()
        self.localNUc = dax.createLocalVec()
        self.localNEc = dax.createLocalVec()
        
        
        # kinetic and external Hamiltonian
        H0.copy(self.H0)
        self.H2p.set(0.)
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1, Vec P, Vec N, Vec NU, Vec NE, Vec U, Vec E, Vec A):
        H1.copy(self.H1h)
        F.copy(self.Fh)
        P.copy(self.Ph)
        N.copy(self.Nh)
        NU.copy(self.NUh)
        NE.copy(self.NEh)
        U.copy(self.Uh)
        E.copy(self.Eh)
        A.copy(self.Ah)
        
    
    def update_external(self, Vec Pext):
        self.H2p.copy(self.H2h)
        self.toolbox.potential_to_hamiltonian(Pext, self.H2p)
        
    
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.mult(X, Y)
        
    
    @cython.boundscheck(False)
    def mult(self, Vec X, Vec Y):
        cdef np.float64_t phisum, phiave
        
        (xs, xe), = self.da2.getRanges()
        
        x  = self.da2.getVecArray(X)
        h1 = self.da1.getVecArray(self.H1p)
        f  = self.da1.getVecArray(self.Fp)
        p  = self.dax.getVecArray(self.Pp)
        n  = self.dax.getVecArray(self.Np)
        nu = self.dax.getVecArray(self.NUp)
        ne = self.dax.getVecArray(self.NEp)
        u  = self.dax.getVecArray(self.Up)
        e  = self.dax.getVecArray(self.Ep)
        a  = self.dax.getVecArray(self.Ap)
        
        f [xs:xe] = x[xs:xe, 0:self.nv]
        p [xs:xe] = x[xs:xe,   self.nv]
        n [xs:xe] = x[xs:xe,   self.nv+1]
        nu[xs:xe] = x[xs:xe,   self.nv+2]
        ne[xs:xe] = x[xs:xe,   self.nv+3]
        u [xs:xe] = x[xs:xe,   self.nv+4]
        e [xs:xe] = x[xs:xe,   self.nv+5]
        
        a[...][:] = 1. / ( e[...] - u[...]**2)
        
        phisum = self.Pp.sum()
        phiave = phisum / self.nx
        
        for j in np.arange(0, self.nv):
            h1[xs:xe, j] = p[xs:xe] - phiave
        
        
        self.matrix_mult(Y)
        
    
    @cython.boundscheck(False)
    cdef matrix_mult(self, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy
        cdef np.uint64_t xe, xs
        
        cdef np.float64_t nmean = self.Np.sum() / self.nx
        
        self.toolbox.compute_density(self.Fp, self.Nc)
        self.toolbox.compute_velocity_density(self.Fp, self.NUc)
        self.toolbox.compute_energy_density(self.Fp, self.NEc)
        
        (xs, xe), = self.da2.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da2.getGlobalArray(Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getLocalArray(self.H0,  self.localH0 )
        cdef np.ndarray[np.float64_t, ndim=2] h1p = self.da1.getLocalArray(self.H1p, self.localH1p)
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
        cdef np.ndarray[np.float64_t, ndim=2] h2p = self.da1.getLocalArray(self.H2p, self.localH2p)
        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
        cdef np.ndarray[np.float64_t, ndim=2] fp  = self.da1.getLocalArray(self.Fp,  self.localFp )
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getLocalArray(self.Fh,  self.localFh )
        
        cdef np.ndarray[np.float64_t, ndim=1] Pp  = self.dax.getLocalArray(self.Pp,  self.localPp)
        cdef np.ndarray[np.float64_t, ndim=1] Np  = self.dax.getLocalArray(self.Np,  self.localNp)
        cdef np.ndarray[np.float64_t, ndim=1] Up  = self.dax.getLocalArray(self.Up,  self.localUp)
        cdef np.ndarray[np.float64_t, ndim=1] Ep  = self.dax.getLocalArray(self.Ep,  self.localEp)
        cdef np.ndarray[np.float64_t, ndim=1] Ap  = self.dax.getLocalArray(self.Ap,  self.localAp)
        
        cdef np.ndarray[np.float64_t, ndim=1] Ph  = self.dax.getLocalArray(self.Ph,  self.localPh)
        cdef np.ndarray[np.float64_t, ndim=1] Nh  = self.dax.getLocalArray(self.Nh,  self.localNh)
        cdef np.ndarray[np.float64_t, ndim=1] Uh  = self.dax.getLocalArray(self.Uh,  self.localUh)
        cdef np.ndarray[np.float64_t, ndim=1] Eh  = self.dax.getLocalArray(self.Eh,  self.localEh)
        cdef np.ndarray[np.float64_t, ndim=1] Ah  = self.dax.getLocalArray(self.Ah,  self.localAh)
        
        cdef np.ndarray[np.float64_t, ndim=1] Nc  = self.dax.getLocalArray(self.Nc,  self.localNc)
        cdef np.ndarray[np.float64_t, ndim=1] Uc  = self.dax.getLocalArray(self.Uc,  self.localUc)
        cdef np.ndarray[np.float64_t, ndim=1] Ec  = self.dax.getLocalArray(self.Ec,  self.localEc)
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h_ave = h0 + 0.5 * (h1p + h1h + h2p + h2h)
        
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Poisson equation
            y[iy, self.nv] = - ( Pp[ix-1] + Pp[ix+1] - 2. * Pp[ix] ) * self.hx2_inv + self.charge * (Np[ix] - nmean)
            
            
            # moments
            y[iy, self.nv+1] = Np [ix] - Nc [ix]
            y[iy, self.nv+2] = NUp[ix] - NUc[ix]
            y[iy, self.nv+3] = NEp[ix] - NEc[ix]
            y[iy, self.nv+4] = Up [ix]
            y[iy, self.nv+5] = Ep [ix]
#            y[iy, self.nv+4] = Up [ix] - NUp[ix] / Np[ix]
#            y[iy, self.nv+5] = Ep [ix] - NEp[ix] / Np[ix]
            
            
            # Vlasov equation
            for j in np.arange(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fp[ix,j]
                    
                else:
                    y[iy, j] = self.toolbox.time_derivative_woa(fp, ix, j) \
                             - self.toolbox.time_derivative_woa(fh, ix, j) \
                             + self.toolbox.arakawa_J4(f_ave, h_ave, ix, j) #\
#                             - 0.5 * self.nu * self.toolbox.collT1woa(fp, Np, Up, Ep, Ap, ix, j) \
#                             - 0.5 * self.nu * self.toolbox.collT1woa(fh, Nh, Uh, Eh, Ah, ix, j) \
#                             - 0.5 * self.nu * self.toolbox.collT2woa(fp, Np, Up, Ep, Ap, ix, j) \
#                             - 0.5 * self.nu * self.toolbox.collT2woa(fh, Nh, Uh, Eh, Ah, ix, j)
