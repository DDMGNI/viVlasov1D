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
        
        self.H2.set(0.)
        
        # create moment vectors
        self.A1d = self.dax.createGlobalVec()
        self.A2d = self.dax.createGlobalVec()
        self.A3d = self.dax.createGlobalVec()
        self.A1p = self.dax.createGlobalVec()
        self.A2p = self.dax.createGlobalVec()
        self.A3p = self.dax.createGlobalVec()
        
        self.Nd = self.dax.createGlobalVec()
        self.Ud = self.dax.createGlobalVec()
        self.Ed = self.dax.createGlobalVec()
        self.Np = self.dax.createGlobalVec()
        self.Up = self.dax.createGlobalVec()
        self.Ep = self.dax.createGlobalVec()
        
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
        
        self.localA1d = dax.createLocalVec()
        self.localA2d = dax.createLocalVec()
        self.localA3d = dax.createLocalVec()
        self.localA1p = dax.createLocalVec()
        self.localA2p = dax.createLocalVec()
        self.localA3p = dax.createLocalVec()
        
        self.localNd  = dax.createLocalVec()
        self.localUd  = dax.createLocalVec()
        self.localEd  = dax.createLocalVec()
        self.localNp  = dax.createLocalVec()
        self.localUp  = dax.createLocalVec()
        self.localEp  = dax.createLocalVec()
        
        # kinetic Hamiltonian
        H0.copy(self.H0)
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1):
        self.H2.copy(self.H2h)
        
        F.copy(self.Fh)
        
        self.H0.copy(self.Hh)
        self.Hh.axpy(1., H1)
        self.Hh.axpy(1., self.H2h)
        
    
    def update_previous(self, Vec F, Vec H1):
        F.copy(self.Fp)
        
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
        
        
        f[xs:xe] = x[xs:xe, 0:self.nv]
        p[xs:xe] = x[xs:xe,   self.nv]
        
        for j in np.arange(0, self.nv):
            h[xs:xe, j] = p[xs:xe]
        
        
        self.update_previous(F, H)
        
        
        
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
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fd  = self.da1.getVecArray(self.localFd)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fp  = self.da1.getVecArray(self.localFp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh)[...]
        cdef np.ndarray[np.float64_t, ndim=2] hd  = self.da1.getVecArray(self.localHd)[...]
        cdef np.ndarray[np.float64_t, ndim=2] hp  = self.da1.getVecArray(self.localHp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] hh  = self.da1.getVecArray(self.localHh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] pd  = self.dax.getVecArray(self.localPd)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h_ave = 0.5 * (hp + hh)
        
        
        # calculate moments
        self.toolbox.collN_moments(dF,      self.A1d, self.A2d, self.A3d, self.Nd, self.Ud, self.Ed)
        self.toolbox.collN_moments(self.Fp, self.A1p, self.A2p, self.A3p, self.Np, self.Up, self.Ep)
        
        self.dax.globalToLocal(self.A1d, self.localA1d)
        self.dax.globalToLocal(self.A2d, self.localA2d)
        self.dax.globalToLocal(self.A3d, self.localA3d)
        self.dax.globalToLocal(self.A1p, self.localA1p)
        self.dax.globalToLocal(self.A2p, self.localA2p)
        self.dax.globalToLocal(self.A3p, self.localA3p)
        
        self.dax.globalToLocal(self.Nd, self.localNd)
        self.dax.globalToLocal(self.Ud, self.localUd)
        self.dax.globalToLocal(self.Ed, self.localEd)
        self.dax.globalToLocal(self.Np, self.localNp)
        self.dax.globalToLocal(self.Up, self.localUp)
        self.dax.globalToLocal(self.Ep, self.localEp)
        
        A1d = self.dax.getVecArray(self.localA1d)[...]
        A2d = self.dax.getVecArray(self.localA2d)[...]
        A3d = self.dax.getVecArray(self.localA3d)[...]
        A1p = self.dax.getVecArray(self.localA1p)[...]
        A2p = self.dax.getVecArray(self.localA2p)[...]
        A3p = self.dax.getVecArray(self.localA3p)[...]
        
        Nd = self.dax.getVecArray(self.localNd)[...]
        Ud = self.dax.getVecArray(self.localUd)[...]
        Ed = self.dax.getVecArray(self.localEd)[...]
        Np = self.dax.getVecArray(self.localNp)[...]
        Up = self.dax.getVecArray(self.localUp)[...]
        Ep = self.dax.getVecArray(self.localEp)[...]
        
        
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
                             - 0.5 * self.nu * self.toolbox.collT1(fd, A1p, A2p, A3p, ix, j) \
                             - 0.5 * self.nu * self.collT1(fp, A1p, A2p, A3p, Nd, Np, Ud, Up, Ed, Ep, ix, j) \
                             - 0.5 * self.nu * self.toolbox.collT2(fd, ix, j)



    cdef np.float64_t collT1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] A1p,
                                   np.ndarray[np.float64_t, ndim=1] A2p,
                                   np.ndarray[np.float64_t, ndim=1] A3p,
                                   np.ndarray[np.float64_t, ndim=1] Nd,
                                   np.ndarray[np.float64_t, ndim=1] Np,
                                   np.ndarray[np.float64_t, ndim=1] Ud,
                                   np.ndarray[np.float64_t, ndim=1] Up,
                                   np.ndarray[np.float64_t, ndim=1] Ed,
                                   np.ndarray[np.float64_t, ndim=1] Ep,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
        
        cdef np.float64_t result = 0.
        
        result += 0.25 * ( \
                           + 1. * Nd[i-1] * ( Np[i-1] * v[j+1] - Up[i-1] ) * f[i-1, j+1] * A3p[i-1] \
                           - 1. * Nd[i-1] * ( Np[i-1] * v[j-1] - Up[i-1] ) * f[i-1, j-1] * A3p[i-1] \
                           + 2. * Nd[i  ] * ( Np[i  ] * v[j+1] - Up[i  ] ) * f[i,   j+1] * A3p[i  ] \
                           - 2. * Nd[i  ] * ( Np[i  ] * v[j-1] - Up[i  ] ) * f[i,   j-1] * A3p[i  ] \
                           + 1. * Nd[i+1] * ( Np[i+1] * v[j+1] - Up[i+1] ) * f[i+1, j+1] * A3p[i+1] \
                           - 1. * Nd[i+1] * ( Np[i+1] * v[j-1] - Up[i+1] ) * f[i+1, j-1] * A3p[i+1] \
                         ) * 0.5 / self.hv
        
        result += 0.25 * ( \
                           + 1. * Np[i-1] * ( Nd[i-1] * v[j+1] - Ud[i-1] ) * f[i-1, j+1] * A3p[i-1] \
                           - 1. * Np[i-1] * ( Nd[i-1] * v[j-1] - Ud[i-1] ) * f[i-1, j-1] * A3p[i-1] \
                           + 2. * Np[i  ] * ( Nd[i  ] * v[j+1] - Ud[i  ] ) * f[i,   j+1] * A3p[i  ] \
                           - 2. * Np[i  ] * ( Nd[i  ] * v[j-1] - Ud[i  ] ) * f[i,   j-1] * A3p[i  ] \
                           + 1. * Np[i+1] * ( Nd[i+1] * v[j+1] - Ud[i+1] ) * f[i+1, j+1] * A3p[i+1] \
                           - 1. * Np[i+1] * ( Nd[i+1] * v[j-1] - Ud[i+1] ) * f[i+1, j-1] * A3p[i+1] \
                        ) * 0.5 / self.hv
        
        result -= 0.25 * ( \
                           + 1. * Np[i-1] * (Np[i-1] * v[j+1] - Up[i-1]) * f[i-1, j+1] * A3p[i-1]**2 * ( Nd[i-1] * Ep[i-1] + Np[i-1] * Ed[i-1] - 2. * Ud[i-1] * Up[i-1] ) \
                           - 1. * Np[i-1] * (Np[i-1] * v[j-1] - Up[i-1]) * f[i-1, j-1] * A3p[i-1]**2 * ( Nd[i-1] * Ep[i-1] + Np[i-1] * Ed[i-1] - 2. * Ud[i-1] * Up[i-1] ) \
                           + 2. * Np[i  ] * (Np[i  ] * v[j+1] - Up[i  ]) * f[i,   j+1] * A3p[i  ]**2 * ( Nd[i  ] * Ep[i  ] + Np[i  ] * Ed[i  ] - 2. * Ud[i  ] * Up[i  ] ) \
                           - 2. * Np[i  ] * (Np[i  ] * v[j-1] - Up[i  ]) * f[i,   j-1] * A3p[i  ]**2 * ( Nd[i  ] * Ep[i  ] + Np[i  ] * Ed[i  ] - 2. * Ud[i  ] * Up[i  ] ) \
                           + 1. * Np[i+1] * (Np[i+1] * v[j+1] - Up[i+1]) * f[i+1, j+1] * A3p[i+1]**2 * ( Nd[i+1] * Ep[i+1] + Np[i+1] * Ed[i+1] - 2. * Ud[i+1] * Up[i+1] ) \
                           - 1. * Np[i+1] * (Np[i+1] * v[j-1] - Up[i+1]) * f[i+1, j-1] * A3p[i+1]**2 * ( Nd[i+1] * Ep[i+1] + Np[i+1] * Ed[i+1] - 2. * Ud[i+1] * Up[i+1] ) \
                        ) * 0.5 / self.hv
        
        return result


