'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa import PETScArakawa


cdef class PETScVlasovFunction(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DA da1, DA dax, Vec H0,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv):
        '''
        Constructor
        '''
        
        assert da1.getDim() == 2
        assert dax.getDim() == 1
        
        # distributed array
        self.da1 = da1
        self.dax = dax
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv
        
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # create hamiltonian and history vectors
        self.Fh  = self.da1.createGlobalVec()
        self.H1h = self.da1.createGlobalVec()
        
        # create local vectors
        self.localB   = da1.createLocalVec()
        self.localF   = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        self.localH0  = da1.createLocalVec()
        self.localH1  = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        
        # create Arakawa solver object
        self.arakawa = PETScArakawa(da1, nx, nv, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1):
        F.copy(self.Fh)
        H1.copy(self.H1h)
        
    
#    def mult(self, Mat mat, Vec X, Vec Y):
#        pass
        
    
#    @cython.boundscheck(False)
    def matrix_mult(self, Vec F, Vec H1, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(F,  self.localF)
        self.da1.globalToLocal(H1, self.localH1)
        
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.H1h, self.localH1h)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da1.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] f   = self.da1.getVecArray(self.localF)  [...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h1  = self.da1.getVecArray(self.localH1) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
            
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = 0.0
                    
                else:
                    y[iy, jy] = self.time_derivative(f,  ix, jx) \
                              - self.time_derivative(fh, ix, jx) \
                              + 0.5 * self.arakawa.arakawa(f,  h0 + h1h, ix, jx) \
                              + 0.5 * self.arakawa.arakawa(fh, h0 + h1,  ix, jx)
                    
        
    
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