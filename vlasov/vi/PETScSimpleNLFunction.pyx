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
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localF  = da1.createLocalVec()
        self.localFh = da1.createLocalVec()
        self.localH  = da1.createLocalVec()
        self.localHh = da1.createLocalVec()
        self.localP  = dax.createLocalVec()
        self.localPh = dax.createLocalVec()
        
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
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        cdef np.float64_t laplace, integral, nmean, phisum
        
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
        cdef np.ndarray[np.float64_t, ndim=2] f  = self.da1.getVecArray(self.localF )[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh = self.da1.getVecArray(self.localFh)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h  = self.da1.getVecArray(self.localH )[...]
        cdef np.ndarray[np.float64_t, ndim=2] hh = self.da1.getVecArray(self.localHh)[...]
        cdef np.ndarray[np.float64_t, ndim=1] p  = self.dax.getVecArray(self.localP )[...]
        cdef np.ndarray[np.float64_t, ndim=1] ph = self.dax.getVecArray(self.localPh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f_ave = 0.5 * (f + fh)
        cdef np.ndarray[np.float64_t, ndim=2] h_ave = 0.5 * (h + hh)
        cdef np.ndarray[np.float64_t, ndim=1] p_ave = 0.5 * (p + ph)
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            
            if i == 0:
#                y[iy, self.nv] = 0.
                y[iy, self.nv] = p[ix]
#                y[iy, self.nv] = phisum
            
            else:
                    
#                laplace  = (p_ave[ix-1] + p_ave[ix+1] - 2. * p_ave[ix]) * self.hx2_inv
                laplace  = (p[ix-1] + p[ix+1] - 2. * p[ix]) * self.hx2_inv
                
                integral = ( \
                             + 1. * f[ix-1, :].sum() \
                             + 2. * f[ix,   :].sum() \
                             + 1. * f[ix+1, :].sum() \
                           ) * 0.25 * self.hv
#                integral = ( \
#                             + 1. * f_ave[ix-1, :].sum() \
#                             + 2. * f_ave[ix,   :].sum() \
#                             + 1. * f_ave[ix+1, :].sum() \
#                           ) * 0.25 * self.hv
                
#                y[iy, self.nv] = - laplace + self.poisson_const * (integral - nmean) #* self.hx2
                y[iy, self.nv] = - laplace + self.poisson_const * (integral - 1.) #* self.hx2
            
            # Vlasov Equation
            for j in np.arange(0, self.nv):
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = f[ix,j]
#                    y[iy, j] = 0.0
                    
                else:
                    y[iy, j] = self.time_derivative(f,  ix, j) \
                             - self.time_derivative(fh, ix, j) \
                             + self.arakawa.arakawa(f_ave, h_ave, ix, j) \
                             - self.alpha * self.coll2(f_ave, ix, j)
#                             - self.alpha * self.coll1(f_ave, ix, j) #\
#                             - self.alpha * self.coll0(f_ave, ix, j) \
                    
                    
    
#    @cython.boundscheck(False)
#    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
#                                            np.uint64_t i, np.uint64_t j):
#        '''
#        Time Derivative
#        '''
#        
#        cdef np.float64_t result
#        
#        result = x[i, j] / self.ht
#        
#        return result


    @cython.boundscheck(False)
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
                                            np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * x[i-1, j-1] \
                   + 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   + 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) / (16. * self.ht)
        
        return result


    @cython.boundscheck(False)
    cdef np.float64_t coll0(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * x[i-1, j-1] \
                   + 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   + 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) / 16.
                 
#        result = ( \
#                   + 1. * x[i,   j-1] \
#                   + 2. * x[i,   j  ] \
#                   + 1. * x[i,   j+1] \
#                 ) * 0.25
        
        return result
        
        
    @cython.boundscheck(False)
    cdef np.float64_t coll1(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
#        result = 3. * ( \
#                   + 1. * ( x[i-1, j  ] - x[i-1, j-1] ) * (self.v[j-1] + self.v[j  ]) \
#                   + 1. * ( x[i-1, j+1] - x[i-1, j  ] ) * (self.v[j  ] + self.v[j+1]) \
#                   + 2. * ( x[i,   j  ] - x[i,   j-1] ) * (self.v[j-1] + self.v[j  ]) \
#                   + 2. * ( x[i,   j+1] - x[i,   j  ] ) * (self.v[j  ] + self.v[j+1]) \
#                   + 1. * ( x[i+1, j  ] - x[i+1, j-1] ) * (self.v[j-1] + self.v[j  ]) \
#                   + 1. * ( x[i+1, j+1] - x[i+1, j  ] ) * (self.v[j  ] + self.v[j+1]) \
#                 ) * 0.25 * 0.25 / self.hv
                 
#        result = 3. * ( \
#                   + ( x[i,   j  ] - x[i,   j-1] ) * (self.v[j-1] + self.v[j  ]) \
#                   + ( x[i,   j+1] - x[i,   j  ] ) * (self.v[j  ] + self.v[j+1]) \
#                 ) * 0.25 / self.hv
        
#        result = 1. * ( \
#                   + 1. * ( x[i-1, j+1] * self.v[j+1] - x[i-1, j-1] * self.v[j-1] ) \
#                   + 2. * ( x[i,   j+1] * self.v[j+1] - x[i,   j-1] * self.v[j-1] ) \
#                   + 1. * ( x[i+1, j+1] * self.v[j+1] - x[i+1, j-1] * self.v[j-1] ) \
#                 ) * 0.25 * 0.5 / self.hv
        
#        result = 3. * ( \
#                   + ( x[i,   j+1] * self.v[j+1] - x[i,   j-1] * self.v[j-1] ) \
#                 ) * 0.5 / self.hv
        
        
        
#        if j == 1:
#            result = ( \
#                       + 1. * ( x[i-1, j  ] + x[i-1, j+1] ) * (self.v[j  ] + self.v[j+1]) \
#                       + 2. * ( x[i,   j  ] + x[i,   j+1] ) * (self.v[j  ] + self.v[j+1]) \
#                       + 1. * ( x[i+1, j  ] + x[i+1, j+1] ) * (self.v[j  ] + self.v[j+1]) \
#                     ) * 0.25 * 0.25 / self.hv
#
###            result = ( \
###                       + 1. * ( x[i-1, j] * self.v[j] - x[i-1, j-1] * self.v[j-1] ) \
###                       + 2. * ( x[i,   j] * self.v[j] - x[i,   j-1] * self.v[j-1] ) \
###                       + 1. * ( x[i+1, j] * self.v[j] - x[i+1, j-1] * self.v[j-1] ) \
###                     ) * 0.25 / self.hv * 0.5
##        
##            result = ( \
##                       + 1. * ( x[i-1, j] * self.v[j] + x[i-1, j+1] * self.v[j+1] ) \
##                       + 2. * ( x[i,   j] * self.v[j] + x[i,   j+1] * self.v[j+1] ) \
##                       + 1. * ( x[i+1, j] * self.v[j] + x[i+1, j+1] * self.v[j+1] ) \
##                     ) * 0.25 / self.hv * 0.5
#        
##        if j == 2:
##
##            result = ( \
##                       + 1. * x[i-1, j+1] * self.v[j+1] \
##                       + 2. * x[i,   j+1] * self.v[j+1] \
##                       + 1. * x[i+1, j+1] * self.v[j+1] \
##                     ) * 0.25 / self.hv * 0.5
#        
#        elif j == self.nv-2:
#            result = ( \
#                       - 1. * ( x[i-1, j-1] + x[i-1, j  ] ) * (self.v[j-1] + self.v[j  ]) \
#                       - 2. * ( x[i,   j-1] + x[i,   j  ] ) * (self.v[j-1] + self.v[j  ]) \
#                       - 1. * ( x[i+1, j-1] + x[i+1, j  ] ) * (self.v[j-1] + self.v[j  ]) \
#                     ) * 0.25 * 0.25 / self.hv
#
###            result = ( \
###                       + 1. * ( x[i-1, j+1] * self.v[j+1] - x[i-1, j] * self.v[j] ) \
###                       + 2. * ( x[i,   j+1] * self.v[j+1] - x[i,   j] * self.v[j] ) \
###                       + 1. * ( x[i+1, j+1] * self.v[j+1] - x[i+1, j] * self.v[j] ) \
###                     ) * 0.25 / self.hv * 0.5
##        
##            result = ( \
##                       - 1. * ( x[i-1, j-1] * self.v[j-1] + x[i-1, j] * self.v[j] ) \
##                       - 2. * ( x[i,   j-1] * self.v[j-1] + x[i,   j] * self.v[j] ) \
##                       - 1. * ( x[i+1, j-1] * self.v[j-1] + x[i+1, j] * self.v[j] ) \
##                     ) * 0.25 / self.hv * 0.5
#                    
##        elif j == self.nv-3:
##            result = ( \
##                       - 1. * x[i-1, j-1] * self.v[j-1] \
##                       - 2. * x[i,   j-1] * self.v[j-1] \
##                       - 1. * x[i+1, j-1] * self.v[j-1] \
##                     ) * 0.25 / self.hv * 0.5
#                    
#        else:
        result = ( \
                   + 1. * ( x[i-1, j  ] + x[i-1, j+1] ) * (self.v[j  ] + self.v[j+1]) \
                   + 2. * ( x[i,   j  ] + x[i,   j+1] ) * (self.v[j  ] + self.v[j+1]) \
                   + 1. * ( x[i+1, j  ] + x[i+1, j+1] ) * (self.v[j  ] + self.v[j+1]) \
                   - 1. * ( x[i-1, j-1] + x[i-1, j  ] ) * (self.v[j-1] + self.v[j  ]) \
                   - 2. * ( x[i,   j-1] + x[i,   j  ] ) * (self.v[j-1] + self.v[j  ]) \
                   - 1. * ( x[i+1, j-1] + x[i+1, j  ] ) * (self.v[j-1] + self.v[j  ]) \
                 ) * 0.25 * 0.25 / self.hv

#            result = ( \
#                       + 1. * ( x[i-1, j+1] * self.v[j+1] - x[i-1, j-1] * self.v[j-1] ) \
#                       + 2. * ( x[i,   j+1] * self.v[j+1] - x[i,   j-1] * self.v[j-1] ) \
#                       + 1. * ( x[i+1, j+1] * self.v[j+1] - x[i+1, j-1] * self.v[j-1] ) \
#                     ) * 0.25 / self.hv * 0.5
        
        
        
        
        return result
    
    
    @cython.boundscheck(False)
    cdef np.float64_t coll2(self, np.ndarray[np.float64_t, ndim=2] x,
                                  np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        
        if j == 1:
            result = ( \
                         + 1. * ( x[i-1, j+1] - x[i-1, j  ] ) \
                         + 2. * ( x[i,   j+1] - x[i,   j  ] ) \
                         + 1. * ( x[i+1, j+1] - x[i+1, j  ] ) \
                     ) * 0.25 * self.hv2_inv
        
        elif j == self.nv-2:
            result = ( \
                         - 1. * ( x[i-1, j  ] - x[i-1, j-1] ) \
                         - 2. * ( x[i,   j  ] - x[i,   j-1] ) \
                         - 1. * ( x[i+1, j  ] - x[i+1, j-1] ) \
                     ) * 0.25 * self.hv2_inv

        else:
            result = ( \
                         + 1. * ( x[i-1, j+1] - x[i-1, j  ] ) \
                         + 2. * ( x[i,   j+1] - x[i,   j  ] ) \
                         + 1. * ( x[i+1, j+1] - x[i+1, j  ] ) \
                         - 1. * ( x[i-1, j  ] - x[i-1, j-1] ) \
                         - 2. * ( x[i,   j  ] - x[i,   j-1] ) \
                         - 1. * ( x[i+1, j  ] - x[i+1, j-1] ) \
                     ) * 0.25 * self.hv2_inv
        
        
#        result = ( \
#                     + 1. * ( x[i-1, j+1] - 2. * x[i-1, j  ] + x[i-1, j-1] ) \
#                     + 2. * ( x[i,   j+1] - 2. * x[i,   j  ] + x[i,   j-1] ) \
#                     + 1. * ( x[i+1, j+1] - 2. * x[i+1, j  ] + x[i+1, j-1] ) \
#                 ) * 0.25 * self.hv2_inv
        
#        result = ( \
#                     + 1. * ( x[i-1, j+1] - x[i-1, j  ] ) * (self.v[j  ] + self.v[j+1])**2 \
#                     - 1. * ( x[i-1, j  ] - x[i-1, j-1] ) * (self.v[j-1] + self.v[j  ])**2 \
#                     + 2. * ( x[i,   j+1] - x[i,   j  ] ) * (self.v[j  ] + self.v[j+1])**2 \
#                     - 2. * ( x[i,   j  ] - x[i,   j-1] ) * (self.v[j-1] + self.v[j  ])**2 \
#                     + 1. * ( x[i+1, j+1] - x[i+1, j  ] ) * (self.v[j  ] + self.v[j+1])**2 \
#                     - 1. * ( x[i+1, j  ] - x[i+1, j-1] ) * (self.v[j-1] + self.v[j  ])**2 \
#                 ) * 0.25 * 0.25 * self.hv2_inv
        
#        result = ( \
#                     + ( x[i,   j-1] - x[i,   j  ] ) * (self.v[j-1] + self.v[j  ])**2 \
#                     + ( x[i,   j+1] - x[i,   j  ] ) * (self.v[j  ] + self.v[j+1])**2 \
#                 ) * 0.25 * self.hv2_inv
        
#        result = ( \
#                     + 1. * x[i-1, j-1] * self.v[j-1]**2 \
#                     - 2. * x[i-1, j  ] * self.v[j  ]**2 \
#                     + 1. * x[i-1, j+1] * self.v[j+1]**2 \
#                     + 2. * x[i,   j-1] * self.v[j-1]**2 \
#                     - 4. * x[i,   j  ] * self.v[j  ]**2 \
#                     + 2. * x[i,   j+1] * self.v[j+1]**2 \
#                     + 1. * x[i+1, j-1] * self.v[j-1]**2 \
#                     - 2. * x[i+1, j  ] * self.v[j  ]**2 \
#                     + 1. * x[i+1, j+1] * self.v[j+1]**2 \
#                 ) * 0.25 * self.hv2_inv
        
#        result = ( \
#                     + 1. * x[i,   j-1] * self.v[j-1]**2 \
#                     - 2. * x[i,   j  ] * self.v[j  ]**2 \
#                     + 1. * x[i,   j+1] * self.v[j+1]**2 \
#                 ) * self.hv2_inv
        
        return result
    
    
    
