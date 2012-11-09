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


cdef class PETScJacobian(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t hx2
    cdef np.float64_t hx2_inv
    
    cdef np.float64_t poisson_const
    
    cdef DA dax
    cdef DA da1
    cdef DA da2
    
    cdef Vec Xh
    cdef Vec Xp
    cdef Vec H0
    cdef Vec F
    cdef Vec PHI
    
    cdef Vec localX
    cdef Vec localXh
    cdef Vec localXp
    cdef Vec localH0
    
    cdef PETScArakawa arakawa
    
    
    def __init__(self, DA da1, DA da2, DA dax, Vec H0,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 np.float64_t poisson_const):
        '''
        Constructor
        '''
        
        assert dax.getDim() == 1
        assert da1.getDim() == 2
        assert da2.getDim() == 2
        
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
        
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # poisson constant
        self.poisson_const = poisson_const
        
        # create history vectors
        self.Xh  = self.da2.createGlobalVec()
        self.Xp  = self.da2.createGlobalVec()
        self.F   = self.da1.createGlobalVec()
        self.PHI = self.dax.createGlobalVec()
        
        # create local vectors
        self.localX  = da2.createLocalVec()
        self.localXh = da2.createLocalVec()
        self.localXp = da2.createLocalVec()
        self.localH0 = da1.createLocalVec()
        
        # create Arakawa solver object
        self.arakawa     = PETScArakawa(da1, nx, nv, hx, hv)
        
    
    def update_history(self, Vec X):
#        X.copy(self.Xh)
        
        x  = self.da2.getVecArray(X)
        xh = self.da2.getVecArray(self.Xh)
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        xh[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        
    
    def update_previous(self, Vec X):
#        X.copy(self.Xp)
        
        x  = self.da2.getVecArray(X)
        xp = self.da2.getVecArray(self.Xp)
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        xp[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        
    
#    @cython.boundscheck(False)
#    def mult(self, SNES snes, Vec X, Vec Y):
    def mult(self, Mat mat, Vec X, Vec Y):
#    def mult(self, Vec X, Vec Y):
        
#        print("   Jacobian.mult()")
        
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        cdef np.float64_t laplace, integral, nmean, phisum
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        self.da2.globalToLocal(X,       self.localX)
        self.da2.globalToLocal(self.Xh, self.localXh)
        self.da2.globalToLocal(self.Xp, self.localXp)
        self.da1.globalToLocal(self.H0, self.localH0)
        
        cdef np.ndarray[np.float64_t, ndim=3] y  = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=3] gx = self.da2.getVecArray(X)[...]
        cdef np.ndarray[np.float64_t, ndim=3] dx = self.da2.getVecArray(self.localX) [...]
        cdef np.ndarray[np.float64_t, ndim=3] xh = self.da2.getVecArray(self.localXh)[...]
        cdef np.ndarray[np.float64_t, ndim=3] xp = self.da2.getVecArray(self.localXp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] df = dx[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] dp = dx[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] fh = xh[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] ph = xh[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] fp = xp[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] pp = xp[:,:,1]
        
        
        cdef np.ndarray[np.float64_t, ndim=1] tp = self.dax.getVecArray(self.PHI)[...]
        cdef np.ndarray[np.float64_t, ndim=2] tf = self.da1.getVecArray(self.F)  [...]
        
        tp[:]   = gx[:, 0, 1]
        tf[:,:] = gx[:, :, 0]
        
        phisum = self.PHI.sum()
        nmean  = self.F.sum() * self.hv / self.nx
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
#            if i == 0:
#                y[iy, :, 1] = phisum
#                
#            else:
            laplace  = (dp[ix-1, 0] + dp[ix+1, 0] - 2. * dp[ix, 0]) * self.hx2_inv
            
            integral = ( \
                         + 1. * df[ix-1, :].sum() \
                         + 2. * df[ix,   :].sum() \
                         + 1. * df[ix+1, :].sum() \
                       ) * 0.25 * self.hv
            
            y[iy, :, 1] = - laplace + self.poisson_const * (integral - nmean) # * self.hx2
            
            # Vlasov Equation
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
                
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, jy, 0] = df[ix, jx]
                    
                else:
                    y[iy, jy, 0] = self.time_derivative(df, ix, jx) \
                                 + 0.5 * self.arakawa.arakawa(df, h0 + ph, ix, jx) \
                                 + 0.5 * self.arakawa.arakawa(fh, dp,      ix, jx)
#                                 + 0.25 * self.arakawa.arakawa(df, pp, ix, jx) \
#                                 + 0.25 * self.arakawa.arakawa(df, ph, ix, jx) \
#                                 + 0.25 * self.arakawa.arakawa(fp, dp, ix, jx) \
#                                 + 0.25 * self.arakawa.arakawa(fh, dp, ix, jx)
            
        


#    @cython.boundscheck(False)
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
                                            np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
#        result = x[i,   j  ] / self.ht
        
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
    
    
    
    def isSparse(self):
        return False
    


