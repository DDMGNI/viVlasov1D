'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa import  PETScArakawa
from vlasov.predictor.PETScArakawa cimport PETScArakawa


cdef class PETScSolver(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t hx2_inv
    
    cdef np.float64_t poisson_const
    
    cdef DA dax
    cdef DA da1
    cdef DA da2
    
    cdef Vec B
    cdef Vec X
    cdef Vec X1
    cdef Vec H0
    cdef Vec PHI
    
    cdef Vec localB
    cdef Vec localX
    cdef Vec localX1
    cdef Vec localH0
    
    cdef np.ndarray tf
    cdef np.ndarray tp
    
    cdef PETScArakawa arakawa
    
    
    def __init__(self, DA da1, DA da2, DA dax, Vec X, Vec B, Vec H0,
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
        
        self.hx2_inv = 1. / hx**2
        
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # poisson constant
        self.poisson_const = poisson_const
        
        # save solution and RHS vector
        self.X = X
        self.B = B
        
        # create history vectors
        self.X1  = self.da2.createGlobalVec()
        self.PHI = self.dax.createGlobalVec()
        
        # create local vectors
        self.localB  = da2.createLocalVec()
        self.localX  = da2.createLocalVec()
        self.localX1 = da2.createLocalVec()
        self.localH0 = da1.createLocalVec()
        
        # create temporary numpy array
        (xs, xe), (ys, ye) = self.da2.getRanges()
        self.tf = np.empty((xe-xs, ye-ys))
        self.tp = np.empty((xe-xs, ye-ys))
        
        # create Arakawa solver object
        self.arakawa     = PETScArakawa(da1, da2, nx, nv, hx, hv)
        
    
    def update_history(self, Vec X):
        x  = self.da2.getVecArray(X)
        x1 = self.da2.getVecArray(self.X1)
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        x1[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        
    
#    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        cdef np.float64_t laplace, integral
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        self.da2.globalToLocal(X,       self.localX)
        self.da2.globalToLocal(self.X1, self.localX1)
        self.da1.globalToLocal(self.H0, self.localH0)
        
        cdef np.ndarray[np.float64_t, ndim=3] y  = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=3] x  = self.da2.getVecArray(self.localX) [...]
        cdef np.ndarray[np.float64_t, ndim=3] xh = self.da2.getVecArray(self.localX1)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f  = x [:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] p  = x [:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] fh = xh[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] ph = xh[:,:,1]
        
        
        phi = self.dax.getVecArray(self.PHI)
        phi[xs:xe] = p[1:-1, 0]
        phisum = self.PHI.sum()
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            if i == 0:
                y[iy, :, 1] = phisum
                
            else:
                laplace  = (p[ix-1, 0] - 2. * p[ix, 0] + p[ix+1, 0]) * self.hx2_inv
                
                integral = ( \
                             + 1. * f[ix-1, :].sum() \
                             + 2. * f[ix,   :].sum() \
                             + 1. * f[ix+1, :].sum() \
                           ) * 0.25 * self.hv
                
                y[iy, :, 1] = laplace - self.poisson_const * integral
            
            # Vlasov Equation
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
            
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, jy, 0] = f[ix, jx]
                    
                else:
                    y[iy, jy, 0] = self.time_derivative(f, ix, jx) \
                                 + 0.5  * self.arakawa.arakawa(f,  h0, ix, jx) \
                                 + 0.25 * self.arakawa.arakawa(f,  p,  ix, jx) \
                                 + 0.25 * self.arakawa.arakawa(f,  ph, ix, jx) \
                                 + 0.25 * self.arakawa.arakawa(fh, p,  ix, jx)
                    
        
    
#    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        self.da2.globalToLocal(self.X1, self.localX1)
        self.da1.globalToLocal(self.H0, self.localH0)
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=3] b  = self.da2.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=3] xh = self.da2.getVecArray(self.localX1)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] fh = xh[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] ph = xh[:,:,1]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            if i == 0:
                b[iy, :, 1] = 0.
            else:
                b[iy, :, 1] = - self.poisson_const
            
            
            # Vlasov equation
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
                
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[iy, jy, 0] = 0.0
                    
                else:
                    b[iy, jy, 0] = self.time_derivative(fh, ix, jx) \
                                 - 0.5  * self.arakawa.arakawa(fh, h0, ix, jx) \
                                 - 0.25 * self.arakawa.arakawa(fh, ph, ix, jx)



#    @cython.boundscheck(False)
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
    
    
    
    def isSparse(self):
        return False
    


