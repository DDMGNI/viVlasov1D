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


cdef class PETScJacobianMatrixFree(object):
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
        
        # charge
        self.charge = charge
        
        # collision frequency
        self.nu = coll_freq
        
        # velocity grid
        self.v = v.copy()
        
        # create work and history vectors
        self.H0  = self.da1.createGlobalVec()
        self.H2  = self.da1.createGlobalVec()
        self.H2h = self.da1.createGlobalVec()
        self.Fp  = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        self.Hp  = self.da1.createGlobalVec()
        self.Hh  = self.da1.createGlobalVec()
        self.Pp  = self.dax.createGlobalVec()
        self.Ph  = self.dax.createGlobalVec()
        
        self.H2.set(0.)
        
        # create moment vectors
        self.A1d = self.dax.createGlobalVec()
        self.A2d = self.dax.createGlobalVec()
        self.A1p = self.dax.createGlobalVec()
        self.A2p = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH2  = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()
        self.localFd  = da1.createLocalVec()
        self.localFp  = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        self.localHd  = da1.createLocalVec()
        self.localHp  = da1.createLocalVec()
        self.localHh  = da1.createLocalVec()
        self.localPd  = dax.createLocalVec()
        self.localPp  = dax.createLocalVec()
        self.localPh  = dax.createLocalVec()
        
        self.localA1d = dax.createLocalVec()
        self.localA2d = dax.createLocalVec()
        self.localA1p = dax.createLocalVec()
        self.localA2p = dax.createLocalVec()
        
        # kinetic Hamiltonian
        H0.copy(self.H0)
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1, Vec P):
        self.H2.copy(self.H2h)
        
        F.copy(self.Fh)
        P.copy(self.Ph)
        
        self.H0.copy(self.Hh)
        self.Hh.axpy(1., H1)
        self.Hp.axpy(1., self.H2h)
        
    
    def update_previous(self, Vec F, Vec H1, Vec P):
        F.copy(self.Fp)
        P.copy(self.Pp)
        
        self.H0.copy(self.Hp)
        self.Hp.axpy(1., H1)
        self.Hp.axpy(1., self.H2)
        
    
    def update_external(self, Vec Pext):
        self.toolbox.potential_to_hamiltonian(Pext, self.H2)
        
    
    def update_previous_X(self, Vec X):
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
            h[xs:xe, j] = p[xs:xe]
        
        
        self.update_previous(F, H, P)
        
        
        
    def mult(self, Mat mat, Vec X, Vec Y):
        (xs, xe), = self.da2.getRanges()
        
        H = self.da1.createGlobalVec()
        F = self.da1.createGlobalVec()
        P = self.dax.createGlobalVec()
        
        x = self.da2.getVecArray(X)
        h = self.da1.getVecArray(H)
        f = self.da1.getVecArray(F)
        p = self.dax.getVecArray(P)
        
        
        f[xs:xe] = x[xs:xe, 0:self.nv]
        p[xs:xe] = x[xs:xe,   self.nv]
        
        for j in np.arange(0, self.nv):
            h[xs:xe, j] = p[xs:xe]
        
        
        self.matrix_mult(F, H, P, Y)
    
    
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec dF, Vec dH, Vec dP, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy
        cdef np.uint64_t xe, xs
        
        cdef np.float64_t laplace, integral, nmean, phisum
        
        nmean  = dF.sum() * self.hv / self.nx
        phisum = dP.sum()
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(dF,       self.localFd)
        self.da1.globalToLocal(self.Fp,  self.localFp)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.da1.globalToLocal(dH,       self.localHd)
        self.da1.globalToLocal(self.Hp,  self.localHp)
        self.da1.globalToLocal(self.Hh,  self.localHh)
        self.dax.globalToLocal(dP,       self.localPd)
        self.dax.globalToLocal(self.Pp,  self.localPp)
        self.dax.globalToLocal(self.Ph,  self.localPh)
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fd  = self.da1.getVecArray(self.localFd)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fp  = self.da1.getVecArray(self.localFp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh)[...]
        cdef np.ndarray[np.float64_t, ndim=2] hd  = self.da1.getVecArray(self.localHd)[...]
        cdef np.ndarray[np.float64_t, ndim=2] hp  = self.da1.getVecArray(self.localHp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] hh  = self.da1.getVecArray(self.localHh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] pd  = self.dax.getVecArray(self.localPd)[...]
        cdef np.ndarray[np.float64_t, ndim=1] pp  = self.dax.getVecArray(self.localPp)[...]
        cdef np.ndarray[np.float64_t, ndim=1] ph  = self.dax.getVecArray(self.localPh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h_ave = 0.5 * (hp + hh)
        
        
        # calculate moments
        self.toolbox.coll_moments(dF,      self.A1d, self.A2d)
        self.toolbox.coll_moments(self.Fp, self.A1p, self.A2p)
        
        self.dax.globalToLocal(self.A1d, self.localA1d)
        self.dax.globalToLocal(self.A2d, self.localA2d)
        self.dax.globalToLocal(self.A1p, self.localA1p)
        self.dax.globalToLocal(self.A2p, self.localA2p)
        
        A1d = self.dax.getVecArray(self.localA1d)[...]
        A2d = self.dax.getVecArray(self.localA2d)[...]
        A1p = self.dax.getVecArray(self.localA1p)[...]
        A2p = self.dax.getVecArray(self.localA2p)[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            if i == 0:
                # pin potential to zero at x[0]
                y[iy, self.nv] = pd[ix]
            
            else:
                    
                laplace  = (pd[ix-1] + pd[ix+1] - 2. * pd[ix]) * self.hx2_inv
                
                integral = ( \
                             + 1. * fd[ix-1, :].sum() \
                             + 2. * fd[ix,   :].sum() \
                             + 1. * fd[ix+1, :].sum() \
                           ) * 0.25 * self.hv
                
                y[iy, self.nv] = - laplace + self.charge * (integral - nmean)
            
            
            # Vlasov Equation
            for j in np.arange(0, self.nv):
                
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fd[ix,j]
                    
                else:
                    y[iy, j] = self.toolbox.time_derivative(fd, ix, j) \
                             + 0.5 * self.toolbox.arakawa(fd, h_ave, ix, j) \
                             + 0.5 * self.toolbox.arakawa(f_ave, hd, ix, j) \
                             - 0.5 * self.nu * self.toolbox.coll1(fd, A1p, ix, j) \
                             - 0.5 * self.nu * self.toolbox.coll1(fp, A1d, ix, j) \
                             - 0.5 * self.nu * self.toolbox.coll2(fd, A2p, ix, j) \
                             - 0.5 * self.nu * self.toolbox.coll2(fp, A2d, ix, j)

