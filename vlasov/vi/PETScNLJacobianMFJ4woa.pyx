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


cdef class PETScJacobianMatrixFree(object):
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
        
        self.H1d = self.da1.createGlobalVec()
        self.Fd  = self.da1.createGlobalVec()
        
        self.H1p = self.da1.createGlobalVec()
        self.H2p = self.da1.createGlobalVec()
        self.Fp  = self.da1.createGlobalVec()
        
        self.H1h = self.da1.createGlobalVec()
        self.H2h = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        
        # create spatial vectors
        self.Pp = self.dax.createGlobalVec()
        self.Np = self.dax.createGlobalVec()
        self.Up = self.dax.createGlobalVec()
        self.Ep = self.dax.createGlobalVec()
        self.Ap = self.dax.createGlobalVec()
        
        self.Ph = self.dax.createGlobalVec()
        self.Nh = self.dax.createGlobalVec()
        self.Uh = self.dax.createGlobalVec()
        self.Eh = self.dax.createGlobalVec()
        self.Ah = self.dax.createGlobalVec()
        
#         self.Pc = self.dax.createGlobalVec()
        self.Nc = self.dax.createGlobalVec()
        self.Uc = self.dax.createGlobalVec()
        self.Ec = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        
        self.localH1d = da1.createLocalVec()
        self.localFd  = da1.createLocalVec()
        
        self.localH1p = da1.createLocalVec()
        self.localH2p = da1.createLocalVec()
        self.localFp  = da1.createLocalVec()
        
        self.localH1h = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        
        # create local spatial vectors
        self.localPd  = dax.createLocalVec()
        self.localNd  = dax.createLocalVec()
        self.localNUd = dax.createLocalVec()
        self.localNEd = dax.createLocalVec()
        self.localUd  = dax.createLocalVec()
        self.localEd  = dax.createLocalVec()
        self.localAd  = dax.createLocalVec()
        
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
        
        # initialise null vector
        self.nvec = self.da2.createGlobalVec()
        self.nvec.set(0.)
        nvec_arr = self.da2.getVecArray(self.nvec)[...]
        nvec_arr[:, self.nv] = 1.  
        self.nvec.normalize()   ### TODO ### Disable ?!? ###
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1):
        F.copy(self.Fh)
        H1.copy(self.H1h)
        
    
    def update_previous(self, Vec X):
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
        
        
    def update_external(self, Vec Pext):
        self.H2p.copy(self.H2h)
        self.toolbox.potential_to_hamiltonian(Pext, self.H2p)
        
    
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.mat_mult(X, Y)
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.mat_mult(X, Y)
        
    
    def mat_mult(self, Vec X, Vec Y):
        cdef np.float64_t phisum, phiave
        
        (xs, xe), = self.da2.getRanges()
        
        x  = self.da2.getVecArray(X)
        h1 = self.da1.getVecArray(self.H1d)
        f  = self.da1.getVecArray(self.Fd)
        p  = self.dax.getVecArray(self.Pd)
        n  = self.dax.getVecArray(self.Nd)
        nu = self.dax.getVecArray(self.NUd)
        ne = self.dax.getVecArray(self.NEd)
        u  = self.dax.getVecArray(self.Ud)
        e  = self.dax.getVecArray(self.Ed)
        a  = self.dax.getVecArray(self.Ad)
        
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
        
        cdef np.float64_t laplace
        
        cdef np.float64_t nmean = self.Nd.sum() / self.nx
        
        self.toolbox.compute_density(self.Fd, self.Nc)
        self.toolbox.compute_velocity_density(self.Fd, self.NUc)
        self.toolbox.compute_energy_density(self.Fd, self.NEc)
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(self.H0,  self.localH0 )
        
        self.da1.globalToLocal(self.H1d, self.localH1d)
        self.da1.globalToLocal(self.Fd,  self.localFd )
        
        self.da1.globalToLocal(self.H1p, self.localH1p)
        self.da1.globalToLocal(self.H2p, self.localH2p)
        self.da1.globalToLocal(self.Fp,  self.localFp )
        
        self.da1.globalToLocal(self.H1h, self.localH1h)
        self.da1.globalToLocal(self.H2h, self.localH2h)
        self.da1.globalToLocal(self.Fh,  self.localFh )
        
        self.dax.globalToLocal(self.Pd,  self.localPd )
        self.dax.globalToLocal(self.Nd,  self.localNd )
        self.dax.globalToLocal(self.NUd, self.localNUd)
        self.dax.globalToLocal(self.NEd, self.localNEd)
        self.dax.globalToLocal(self.Ud,  self.localUd )
        self.dax.globalToLocal(self.Ed,  self.localEd )
        self.dax.globalToLocal(self.Ad,  self.localAd )
        
        self.dax.globalToLocal(self.Pp,  self.localPp )
        self.dax.globalToLocal(self.Np,  self.localNp )
        self.dax.globalToLocal(self.NUp, self.localNUp)
        self.dax.globalToLocal(self.NEp, self.localNEp)
        self.dax.globalToLocal(self.Up,  self.localUp )
        self.dax.globalToLocal(self.Ep,  self.localEp )
        self.dax.globalToLocal(self.Ap,  self.localAp )
        
        self.dax.globalToLocal(self.Ph,  self.localPh )
        self.dax.globalToLocal(self.Nh,  self.localNh )
        self.dax.globalToLocal(self.NUh, self.localNUh)
        self.dax.globalToLocal(self.NEh, self.localNEh)
        self.dax.globalToLocal(self.Uh,  self.localUh )
        self.dax.globalToLocal(self.Eh,  self.localEh )
        self.dax.globalToLocal(self.Ah,  self.localAh )
        
        self.dax.globalToLocal(self.Nc,  self.localNc )
        self.dax.globalToLocal(self.NUc, self.localNUc)
        self.dax.globalToLocal(self.NEc, self.localNEc)
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0 )[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h1d = self.da1.getVecArray(self.localH1d)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fd  = self.da1.getVecArray(self.localFd )[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h1p = self.da1.getVecArray(self.localH1p)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2p = self.da1.getVecArray(self.localH2p)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fp  = self.da1.getVecArray(self.localFp )[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getVecArray(self.localH2h)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Pd  = self.dax.getVecArray(self.localPd )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Nd  = self.dax.getVecArray(self.localNd )[...]
        cdef np.ndarray[np.float64_t, ndim=1] NUd = self.dax.getVecArray(self.localNUd)[...]
        cdef np.ndarray[np.float64_t, ndim=1] NEd = self.dax.getVecArray(self.localNEd)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ud  = self.dax.getVecArray(self.localUd )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ed  = self.dax.getVecArray(self.localEd )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ad  = self.dax.getVecArray(self.localAd )[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Pp  = self.dax.getVecArray(self.localPp )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Np  = self.dax.getVecArray(self.localNp )[...]
        cdef np.ndarray[np.float64_t, ndim=1] NUp = self.dax.getVecArray(self.localNUp)[...]
        cdef np.ndarray[np.float64_t, ndim=1] NEp = self.dax.getVecArray(self.localNEp)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Up  = self.dax.getVecArray(self.localUp )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ep  = self.dax.getVecArray(self.localEp )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ap  = self.dax.getVecArray(self.localAp )[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Ph  = self.dax.getVecArray(self.localPh )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Nh  = self.dax.getVecArray(self.localNh )[...]
        cdef np.ndarray[np.float64_t, ndim=1] NUh = self.dax.getVecArray(self.localNUh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] NEh = self.dax.getVecArray(self.localNEh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Uh  = self.dax.getVecArray(self.localUh )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Eh  = self.dax.getVecArray(self.localEh )[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ah  = self.dax.getVecArray(self.localAh )[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Nc  = self.dax.getVecArray(self.localNc )[...]
        cdef np.ndarray[np.float64_t, ndim=1] NUc = self.dax.getVecArray(self.localNUc)[...]
        cdef np.ndarray[np.float64_t, ndim=1] NEc = self.dax.getVecArray(self.localNEc)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h_ave = h0 + 0.5 * (h1p + h1h + h2p + h2h)
        cdef np.ndarray[np.float64_t, ndim=2] hp = h0 + h1p + h2p
        
        cdef np.float64_t coll1_fac = - 0.5 * self.nu * 0.5 / self.hv
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Poisson equation
            y[iy, self.nv] = - ( Pd[ix-1] + Pd[ix+1] - 2. * Pd[ix] ) * self.hx2_inv + self.charge * Nd[ix]
            
            
            # moments
            y[iy, self.nv+1] = Nd [ix] - Nc[ix]
            y[iy, self.nv+2] = NUd[ix] - NUc[ix]
            y[iy, self.nv+3] = NEd[ix] - NEc[ix]
            y[iy, self.nv+4] = Ud [ix] - NUd[ix] / Np[ix] + Nd[ix] * NUp[ix] / Np[ix]**2
            y[iy, self.nv+5] = Ed [ix] - NEd[ix] / Np[ix] + Nd[ix] * NEp[ix] / Np[ix]**2
            
            
            # Vlasov equation
            for j in np.arange(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fd[ix,j]
                    
                else:
                    y[iy, j] = self.toolbox.time_derivative_woa(fp, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(fp, h_ave, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(f_ave, h1p, ix, j) \
                             - 0.5 * self.nu * self.toolbox.collT1woa(fp, Np, Up, Ep, Ap, ix, j) \
                             - 0.5 * self.nu * self.toolbox.collT2woa(fp, Np, Up, Ep, Ap, ix, j)
            
