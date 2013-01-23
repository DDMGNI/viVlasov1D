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

from vlasov.predictor.PETScArakawa import  PETScArakawa
from vlasov.predictor.PETScArakawa cimport PETScArakawa


cdef class PETScFunction(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    
    def __init__(self, DA da1, DA da2, DA dax, Vec H0,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 np.float64_t poisson_const, np.float64_t alpha=0.):
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
        self.poisson_const = poisson_const
        
        # collision constant
        self.alpha = alpha
        
        # velocity grid
        self.v = v.copy()
        
        # create work and history vectors
        self.H0 = self.da1.createGlobalVec()
        self.Fh = self.da1.createGlobalVec()
        self.Hh = self.da1.createGlobalVec()
        self.Ph = self.dax.createGlobalVec()
        
        # create moment vectors
        self.A1p   = self.dax.createGlobalVec()
        self.A2p   = self.dax.createGlobalVec()
        self.A1h   = self.dax.createGlobalVec()
        self.A2h   = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localF  = da1.createLocalVec()
        self.localFh = da1.createLocalVec()
        self.localH  = da1.createLocalVec()
        self.localHh = da1.createLocalVec()
        self.localP  = dax.createLocalVec()
        self.localPh = dax.createLocalVec()
        
        self.localA1p    = dax.createLocalVec()
        self.localA2p    = dax.createLocalVec()
        self.localA1h    = dax.createLocalVec()
        self.localA2h    = dax.createLocalVec()
        
        # kinetic Hamiltonian
        H0.copy(self.H0)
        
        # create Arakawa solver object
        self.arakawa     = PETScArakawa(da1, nx, nv, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1, Vec P):
        F.copy(self.Fh)
        P.copy(self.Ph)
        
        self.H0.copy(self.Hh)
        self.Hh.axpy(1., H1)
        
    
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
        
        cdef np.float64_t laplace, integral, nmean, phisum, denom_p, denom_h
        
        nmean  = F.sum() * self.hv / self.nx
#        nmean += self.Fh.sum() * self.hv / self.nx
#        nmean *= .5

        phisum = P.sum()
        
        (xs, xe), = self.da2.getRanges()
        
        self.da1.globalToLocal(F,       self.localF )
        self.da1.globalToLocal(self.Fh, self.localFh)
        self.da1.globalToLocal(H,       self.localH )
        self.da1.globalToLocal(self.Hh, self.localHh)
        self.dax.globalToLocal(P,       self.localP )
        self.dax.globalToLocal(self.Ph, self.localPh)
        
        cdef np.ndarray[np.float64_t, ndim=2] y  = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] fp = self.da1.getVecArray(self.localF )[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh = self.da1.getVecArray(self.localFh)[...]
        cdef np.ndarray[np.float64_t, ndim=2] hp = self.da1.getVecArray(self.localH )[...]
        cdef np.ndarray[np.float64_t, ndim=2] hh = self.da1.getVecArray(self.localHh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] p  = self.dax.getVecArray(self.localP )[...]
        cdef np.ndarray[np.float64_t, ndim=1] ph = self.dax.getVecArray(self.localPh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h_ave = 0.5 * (hp + hh)
        cdef np.ndarray[np.float64_t, ndim=1] p_ave = 0.5 * (p  + ph)
        
        
        # calculate moments
        cdef np.ndarray[np.float64_t, ndim=1] A1p = self.dax.getVecArray(self.A1p)[...]
        cdef np.ndarray[np.float64_t, ndim=1] A2p = self.dax.getVecArray(self.A2p)[...]
        cdef np.ndarray[np.float64_t, ndim=1] A1h = self.dax.getVecArray(self.A1h)[...]
        cdef np.ndarray[np.float64_t, ndim=1] A2h = self.dax.getVecArray(self.A2h)[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] mom_np = np.zeros_like(A1p)
        cdef np.ndarray[np.float64_t, ndim=1] mom_nh = np.zeros_like(A1h)
        cdef np.ndarray[np.float64_t, ndim=1] mom_up = np.zeros_like(A1p)
        cdef np.ndarray[np.float64_t, ndim=1] mom_uh = np.zeros_like(A1h)
        cdef np.ndarray[np.float64_t, ndim=1] mom_ep = np.zeros_like(A1p)
        cdef np.ndarray[np.float64_t, ndim=1] mom_eh = np.zeros_like(A1h)
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
#            mom_np[iy] = 0.
#            mom_nh[iy] = 0.
#            mom_up[iy] = 0.
#            mom_uh[iy] = 0.
#            mom_ep[iy] = 0.
#            mom_eh[iy] = 0.
#            
#            for j in np.arange(0, (self.nv-1)/2):
#                mom_np[iy] += fp[ix, j] + fp[ix, self.nv-1-j]
#                mom_nh[iy] += fh[ix, j] + fh[ix, self.nv-1-j]
#                mom_up[iy] += self.v[j]    * fp[ix, j]
#                mom_uh[iy] += self.v[j]    * fh[ix, j]
#                mom_ep[iy] += self.v[j]**2 * fp[ix, j]
#                mom_eh[iy] += self.v[j]**2 * fh[ix, j]
#                mom_up[iy] += self.v[self.nv-1-j]    * fp[ix, self.nv-1-j]
#                mom_uh[iy] += self.v[self.nv-1-j]    * fh[ix, self.nv-1-j]
#                mom_ep[iy] += self.v[self.nv-1-j]**2 * fp[ix, self.nv-1-j]
#                mom_eh[iy] += self.v[self.nv-1-j]**2 * fh[ix, self.nv-1-j]
#
#            mom_up[iy] += self.v[(self.nv-1)/2]    * fp[ix, (self.nv-1)/2]
#            mom_uh[iy] += self.v[(self.nv-1)/2]    * fh[ix, (self.nv-1)/2]
#            mom_ep[iy] += self.v[(self.nv-1)/2]**2 * fp[ix, (self.nv-1)/2]
#            mom_eh[iy] += self.v[(self.nv-1)/2]**2 * fh[ix, (self.nv-1)/2]
#                
#            mom_np[iy] *= self.hv
#            mom_nh[iy] *= self.hv
#            mom_up[iy] *= self.hv / mom_np[iy]
#            mom_uh[iy] *= self.hv / mom_nh[iy]
#            mom_ep[iy] *= self.hv
#            mom_eh[iy] *= self.hv
            
            mom_np[iy] = fp[ix, :].sum() * self.hv
            mom_nh[iy] = fh[ix, :].sum() * self.hv
            mom_up[iy] = ( self.v    * fp[ix, :] ).sum() * self.hv / mom_np[iy]
            mom_uh[iy] = ( self.v    * fh[ix, :] ).sum() * self.hv / mom_nh[iy]
            mom_ep[iy] = ( self.v**2 * fp[ix, :] ).sum() * self.hv
            mom_eh[iy] = ( self.v**2 * fh[ix, :] ).sum() * self.hv
            
            denom_p = mom_np[iy] * mom_up[iy]**2 - mom_ep[iy]  
            denom_h = mom_nh[iy] * mom_uh[iy]**2 - mom_eh[iy]  
            
            A1p[iy] = + mom_np[iy] * mom_up[iy] / denom_p
            A1h[iy] = + mom_nh[iy] * mom_uh[iy] / denom_h
            A2p[iy] = - mom_np[iy] / denom_p
            A2h[iy] = - mom_nh[iy] / denom_h
        
        
        self.dax.globalToLocal(self.A1p, self.localA1p)
        self.dax.globalToLocal(self.A2p, self.localA2p)
        self.dax.globalToLocal(self.A1h, self.localA1h)
        self.dax.globalToLocal(self.A2h, self.localA2h)
        
        A1p = self.dax.getVecArray(self.localA1p)[...]
        A2p = self.dax.getVecArray(self.localA2p)[...]
        A1h = self.dax.getVecArray(self.localA1h)[...]
        A2h = self.dax.getVecArray(self.localA2h)[...]
        
#        cdef np.ndarray[np.float64_t, ndim=1] A1 = 0.5 * (A1p + A1h)
#        cdef np.ndarray[np.float64_t, ndim=1] A2 = 0.5 * (A2p + A2h)

        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            
            if i == 0:
#                y[iy, self.nv] = 0.
                y[iy, self.nv] = p[ix]
#                y[iy, self.nv] = phisum
            
            else:
                    
                laplace  = (p[ix-1] + p[ix+1] - 2. * p[ix]) * self.hx2_inv
                
                integral = ( \
                             + 1. * fp[ix-1, :].sum() \
                             + 2. * fp[ix,   :].sum() \
                             + 1. * fp[ix+1, :].sum() \
                           ) * 0.25 * self.hv
                
#                y[iy, self.nv] = - laplace + self.poisson_const * (integral - nmean) #* self.hx2
                y[iy, self.nv] = - laplace + self.poisson_const * (integral - 1.) #* self.hx2
            
            # Vlasov Equation
            for j in np.arange(0, self.nv):
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fp[ix,j]
                    
                else:
                    y[iy, j] = self.time_derivative(fp, ix, j) \
                             - self.time_derivative(fh, ix, j) \
                             + self.arakawa.arakawa(f_ave, h_ave, ix, j) \
                             - 0.5 * self.alpha * self.coll0(fp, A1p, ix, j) \
                             - 0.5 * self.alpha * self.coll0(fh, A1h, ix, j) \
                             - 0.5 * self.alpha * self.coll1(fp, A2p, ix, j) \
                             - 0.5 * self.alpha * self.coll1(fh, A2h, ix, j) \
                             - self.alpha * self.coll2(f_ave, ix, j)

#                             - self.alpha * self.coll0(f_ave, A1, ix, j) \
#                             - self.alpha * self.coll1(f_ave, A2, ix, j) \
#                             - self.alpha * self.coll2(f_ave, ix, j)
    

    @cython.boundscheck(False)
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] f,
                                            np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * f[i-1, j-1] \
                   + 2. * f[i-1, j  ] \
                   + 1. * f[i-1, j+1] \
                   + 2. * f[i,   j-1] \
                   + 4. * f[i,   j  ] \
                   + 2. * f[i,   j+1] \
                   + 1. * f[i+1, j-1] \
                   + 2. * f[i+1, j  ] \
                   + 1. * f[i+1, j+1] \
                 ) / (16. * self.ht)
        
        return result



#    @cython.boundscheck(False)
#    cdef np.float64_t coll0(self, np.ndarray[np.float64_t, ndim=2] f,
#                                  np.ndarray[np.float64_t, ndim=1] A1,
#                                  np.uint64_t i, np.uint64_t j):
#        '''
#        Collision Operator
#        '''
#        
#        cdef np.float64_t result
#        
#        # A1 * df/dv
#        result = 0.25 * ( f[i,   j+1] - f[i,   j-1] ) * A1[i] ) * 0.5 / self.hv
#        
#        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t coll0(self, np.ndarray[np.float64_t, ndim=2] f,
                                  np.ndarray[np.float64_t, ndim=1] A1,
                                  np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        cdef np.float64_t result
        
        # A1 * df/dv
        result = 0.25 * ( \
                          + 1. * ( f[i-1, j+1] - f[i-1, j-1] ) * A1[i-1] \
                          + 2. * ( f[i,   j+1] - f[i,   j-1] ) * A1[i  ] \
                          + 1. * ( f[i+1, j+1] - f[i+1, j-1] ) * A1[i+1] \
                        ) * 0.5 / self.hv
        
        return result
    
    
    
#    @cython.boundscheck(False)
#    cdef np.float64_t coll1(self, np.ndarray[np.float64_t, ndim=2] f,
#                                  np.ndarray[np.float64_t, ndim=1] A2,
#                                  np.uint64_t i, np.uint64_t j):
#        '''
#        Collision Operator
#        '''
#        
#        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
#        
#        cdef np.float64_t result
#        
#        # d/dv ( v * A2 * f )
#        result = ( v[j+1] * f[i,   j+1] - v[j-1] * f[i,   j-1] ) * A2[i  ] ) * 0.5 / self.hv

    
    
    @cython.boundscheck(False)
    cdef np.float64_t coll1(self, np.ndarray[np.float64_t, ndim=2] f,
                                  np.ndarray[np.float64_t, ndim=1] A2,
                                  np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
        
        cdef np.float64_t result
        
        # d/dv ( v * A2 * f )
        result = 0.25 * ( \
                          + 1. * ( v[j+1] * f[i-1, j+1] - v[j-1] * f[i-1, j-1] ) * A2[i-1] \
                          + 2. * ( v[j+1] * f[i,   j+1] - v[j-1] * f[i,   j-1] ) * A2[i  ] \
                          + 1. * ( v[j+1] * f[i+1, j+1] - v[j-1] * f[i+1, j-1] ) * A2[i+1] \
                        ) * 0.5 / self.hv
        
        return result
    
    
    
#    @cython.boundscheck(False)
#    cdef np.float64_t coll2(self, np.ndarray[np.float64_t, ndim=2] f,
#                                  np.uint64_t i, np.uint64_t j):
#        '''
#        Time Derivative
#        '''
#        
#        cdef np.float64_t result
#        
#        result = ( f[i, j+1] - 2. * f[i, j] + f[i, j-1] ) * self.hv2_inv
#        
#        return result

    
    
    @cython.boundscheck(False)
    cdef np.float64_t coll2(self, np.ndarray[np.float64_t, ndim=2] f,
                                  np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * ( f[i-1, j+1] - 2. * f[i-1, j  ] + f[i-1, j-1] ) \
                     + 2. * ( f[i,   j+1] - 2. * f[i,   j  ] + f[i,   j-1] ) \
                     + 1. * ( f[i+1, j+1] - 2. * f[i+1, j  ] + f[i+1, j-1] ) \
                 ) * 0.25 * self.hv2_inv
        
        return result
    
    
    
