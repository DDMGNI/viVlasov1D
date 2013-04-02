'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport DA, SNES, Mat, Vec

from vlasov.vi.Toolbox import Toolbox


cdef class PETScFunction(object):
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
        self.H2  = self.da1.createGlobalVec()
        self.H2h = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        self.Hh  = self.da1.createGlobalVec()
        
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
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH2  = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()
        self.localF   = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        self.localH   = da1.createLocalVec()
        self.localHh  = da1.createLocalVec()
        
        # create local spatial vectors
        self.localPp = dax.createLocalVec()
        self.localNp = dax.createLocalVec()
        self.localUp = dax.createLocalVec()
        self.localEp = dax.createLocalVec()
        self.localAp = dax.createLocalVec()
        
        self.localPh = dax.createLocalVec()
        self.localNh = dax.createLocalVec()
        self.localUh = dax.createLocalVec()
        self.localEh = dax.createLocalVec()
        self.localAh = dax.createLocalVec()
        
        
        # kinetic and external Hamiltonian
        H0.copy(self.H0)
        self.H2.set(0.)
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1, Vec P, Vec N, Vec U, Vec E, Vec A):
        F.copy(self.Fh)
        P.copy(self.Ph)
        N.copy(self.Nh)
        U.copy(self.Uh)
        E.copy(self.Eh)
        A.copy(self.Ah)
        
        self.H0.copy(self.Hh)
        self.Hh.axpy(1., H1)
        
    
    def update_external(self, Vec Pext):
        self.H2.copy(self.H2h)
        self.toolbox.potential_to_hamiltonian(Pext, self.H2)
        
    
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.mult(X, Y)
        
    
    @cython.boundscheck(False)
    def mult(self, Vec X, Vec Y):
        (xs, xe), = self.da2.getRanges()
        
        H = self.da1.createGlobalVec()
        F = self.da1.createGlobalVec()
        
        x = self.da2.getVecArray(X)
        h = self.da1.getVecArray(H)
        f = self.da1.getVecArray(F)
        
        p = self.dax.getVecArray(self.Pp)
        n = self.dax.getVecArray(self.Np)
        u = self.dax.getVecArray(self.Up)
        e = self.dax.getVecArray(self.Ep)
        a = self.dax.getVecArray(self.Ap)
        
        h0 = self.da1.getVecArray(self.H0)
        
        
        f[xs:xe] = x[xs:xe, 0:self.nv]
        p[xs:xe] = x[xs:xe,   self.nv]
        n[xs:xe] = x[xs:xe,   self.nv+1]
        u[xs:xe] = x[xs:xe,   self.nv+2]
        e[xs:xe] = x[xs:xe,   self.nv+3]
        a[xs:xe] = x[xs:xe,   self.nv+4]
        
        for j in np.arange(0, self.nv):
            h[xs:xe, j] = h0[xs:xe, j] + p[xs:xe]
        
        
        self.matrix_mult(F, H, Y)
        
        del F, H
        
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec F, Vec H, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy
        cdef np.uint64_t xe, xs
        
        cdef np.float64_t laplace, integral, nmean, phisum
        
        nmean  = self.Np.sum() / self.nx
        phisum = self.Pp.sum()
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(F,        self.localF )
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.da1.globalToLocal(H,        self.localH )
        self.da1.globalToLocal(self.Hh,  self.localHh)
        self.da1.globalToLocal(self.H2,  self.localH2 )
        self.da1.globalToLocal(self.H2h, self.localH2h)
        
        self.dax.globalToLocal(self.Pp,  self.localPp)
        self.dax.globalToLocal(self.Np,  self.localNp)
        self.dax.globalToLocal(self.Up,  self.localUp)
        self.dax.globalToLocal(self.Ep,  self.localEp)
        self.dax.globalToLocal(self.Ap,  self.localAp)
        
        self.dax.globalToLocal(self.Ph,  self.localPh)
        self.dax.globalToLocal(self.Nh,  self.localNh)
        self.dax.globalToLocal(self.Uh,  self.localUh)
        self.dax.globalToLocal(self.Eh,  self.localEh)
        self.dax.globalToLocal(self.Ah,  self.localAh)
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fp  = self.da1.getVecArray(self.localF  )[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        cdef np.ndarray[np.float64_t, ndim=2] hp  = self.da1.getVecArray(self.localH  )[...]
        cdef np.ndarray[np.float64_t, ndim=2] hh  = self.da1.getVecArray(self.localHh )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2  = self.da1.getVecArray(self.localH2 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getVecArray(self.localH2h)[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Pp = self.dax.getVecArray(self.localPp)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Np = self.dax.getVecArray(self.localNp)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Up = self.dax.getVecArray(self.localUp)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ep = self.dax.getVecArray(self.localEp)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ap = self.dax.getVecArray(self.localAp)[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] Ph = self.dax.getVecArray(self.localPh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Nh = self.dax.getVecArray(self.localNh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Uh = self.dax.getVecArray(self.localUh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Eh = self.dax.getVecArray(self.localEh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] Ah = self.dax.getVecArray(self.localAh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h_ave = 0.5 * (hp + hh + h2 + h2h)
        
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            if i == 0:
                y[iy, self.nv] = Pp[ix]
            
            else:
                    
                laplace  = (Pp[ix-1] + Pp[ix+1] - 2. * Pp[ix]) * self.hx2_inv
                integral = 0.25 * ( 1. * Np[ix-1] + 2. * Np[ix  ] + 1. * Np[ix+1] )
                
                y[iy, self.nv] = - laplace + self.charge * (integral - nmean)
            
            
            # moments
            y[iy, self.nv+1] = self.hv * fp[ix].sum()
            y[iy, self.nv+2] = self.hv * (fp[ix, :] * self.v).sum()
            y[iy, self.nv+3] = self.hv * (fp[ix, :] * self.v * self.v).sum()
            y[iy, self.nv+4] = Np[ix] / (Np[ix] * Ep[ix] - Up[ix] * Up[ix])
            
            
            # Vlasov equation
            for j in np.arange(0, self.nv):
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fp[ix,j]
                    
                else:
                    y[iy, j] = self.toolbox.time_derivative(fp, ix, j) \
                             - self.toolbox.time_derivative(fh, ix, j) \
                             + self.toolbox.arakawa(f_ave, h_ave, ix, j) \
                             - 0.5 * self.nu * self.toolbox.collT1(fp, Np, Up, Ep, Ap, ix, j) \
                             - 0.5 * self.nu * self.toolbox.collT1(fh, Nh, Uh, Eh, Ah, ix, j) \
                             - self.nu * self.toolbox.collT2(f_ave, ix, j)

