'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa import PETScArakawa


cdef class PETScVlasovSolver(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DA da1, DA da2, Vec H0,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv):
        '''
        Constructor
        '''
        
        assert da1.getDim() == 2
        assert da2.getDim() == 2
        
        # distributed array
        self.da1 = da1
        self.da2 = da2
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv
        
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # create history vectors
        self.X0  = self.da2.createGlobalVec()
        self.X1  = self.da2.createGlobalVec()
        
        # create local vectors
        self.localB  = da1.createLocalVec()
        self.localX  = da1.createLocalVec()
        
        self.localH0 = da1.createLocalVec()
        self.localX0 = da2.createLocalVec()
        self.localX1 = da2.createLocalVec()
        
        # create Arakawa solver object
        self.arakawa = PETScArakawa(da1, da2, nx, nv, hx, hv)
        
    
    def update_current(self, Vec X):
        x  = self.da2.getVecArray(X)
        x0 = self.da2.getVecArray(self.X0)
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        x0[xs:xe, ys:ye, :] = x[xs:xe, ys:ye, :]
        
    
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
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X,       self.localX)
        
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da2.globalToLocal(self.X0, self.localX0)
        self.da2.globalToLocal(self.X1, self.localX1)
        
        cdef np.ndarray[np.float64_t, ndim=2] y  = self.da1.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] f  = self.da1.getVecArray(self.localX) [...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        cdef np.ndarray[np.float64_t, ndim=3] x0 = self.da2.getVecArray(self.localX0)[...]
        cdef np.ndarray[np.float64_t, ndim=3] x1 = self.da2.getVecArray(self.localX1)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] fh = x1[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] p  = x0[:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] ph = x1[:,:,1]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
            
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = f[ix, jx]
                    
                else:
                    y[iy, jy] = self.time_derivative(f, ix, jx) \
                              + 0.5 * self.arakawa.arakawa(f,  h0 + ph, ix, jx) \
                              + 0.5 * self.arakawa.arakawa(fh, h0 + p,  ix, jx)
                    
        
    
#    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da2.globalToLocal(self.X1, self.localX1)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] b  = self.da1.getVecArray(B)[...]
        
        cdef np.ndarray[np.float64_t, ndim=3] xh = self.da2.getVecArray(self.localX1)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] fh = xh[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] ph = xh[:,:,1]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
                
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[iy, jy] = 0.0
                    
                else:
                    b[iy, jy] = self.time_derivative(fh, ix, jx)
    


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
