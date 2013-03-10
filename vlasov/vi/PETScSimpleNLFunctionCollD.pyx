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
        self.Ph  = self.dax.createGlobalVec()
        self.H2.set(0.)
        
        # create moment vectors
        self.A1p = self.dax.createGlobalVec()
        self.A2p = self.dax.createGlobalVec()
        self.A3p = self.dax.createGlobalVec()
        self.A1h = self.dax.createGlobalVec()
        self.A2h = self.dax.createGlobalVec()
        self.A3h = self.dax.createGlobalVec()
        
        self.Np = self.dax.createGlobalVec()
        self.Up = self.dax.createGlobalVec()
        self.Ep = self.dax.createGlobalVec()
        self.Nh = self.dax.createGlobalVec()
        self.Uh = self.dax.createGlobalVec()
        self.Eh = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH2  = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()
        self.localF  = da1.createLocalVec()
        self.localFh = da1.createLocalVec()
        self.localH  = da1.createLocalVec()
        self.localHh = da1.createLocalVec()
        self.localP  = dax.createLocalVec()
        self.localPh = dax.createLocalVec()
        
        self.localA1p    = dax.createLocalVec()
        self.localA2p    = dax.createLocalVec()
        self.localA3p    = dax.createLocalVec()
        self.localA1h    = dax.createLocalVec()
        self.localA2h    = dax.createLocalVec()
        self.localA3h    = dax.createLocalVec()
        
        # kinetic Hamiltonian
        H0.copy(self.H0)
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1, Vec P):
        F.copy(self.Fh)
        P.copy(self.Ph)
        
        self.H0.copy(self.Hh)
        self.Hh.axpy(1., H1)
        
    
    def update_external(self, Vec Pext):
        self.H2.copy(self.H2h)
        self.toolbox.potential_to_hamiltonian(Pext, self.H2)
        
    
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
        self.da1.globalToLocal(self.H2,  self.localH2 )
        self.da1.globalToLocal(self.H2h, self.localH2h)
        self.dax.globalToLocal(P,        self.localP )
        self.dax.globalToLocal(self.Ph,  self.localPh)
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fp  = self.da1.getVecArray(self.localF  )[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        cdef np.ndarray[np.float64_t, ndim=2] hp  = self.da1.getVecArray(self.localH  )[...]
        cdef np.ndarray[np.float64_t, ndim=2] hh  = self.da1.getVecArray(self.localHh )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2  = self.da1.getVecArray(self.localH2 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h2h = self.da1.getVecArray(self.localH2h)[...]
        cdef np.ndarray[np.float64_t, ndim=1] p   = self.dax.getVecArray(self.localP  )[...]
        cdef np.ndarray[np.float64_t, ndim=1] ph  = self.dax.getVecArray(self.localPh )[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h_ave = 0.5 * (hp + hh + h2 + h2h)
        cdef np.ndarray[np.float64_t, ndim=1] p_ave = 0.5 * (p  + ph)
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            
            if i == 0:
                y[iy, self.nv] = p[ix]
            
            else:
                    
                laplace  = (p[ix-1] + p[ix+1] - 2. * p[ix]) * self.hx2_inv
                
                integral = ( \
                             + 1. * fp[ix-1, :].sum() \
                             + 2. * fp[ix,   :].sum() \
                             + 1. * fp[ix+1, :].sum() \
                           ) * 0.25 * self.hv
                
                y[iy, self.nv] = - laplace + self.charge * (integral - nmean)
            
            # Vlasov Equation
            for j in np.arange(0, self.nv):
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fp[ix,j]
                    
                else:
                    y[iy, j] = self.toolbox.time_derivative(fp, ix, j) \
                             - self.toolbox.time_derivative(fh, ix, j) \
                             + self.toolbox.arakawa(f_ave, h_ave, ix, j) \
                             - self.nu * self.toolbox.collT2(f_ave, ix, j)

