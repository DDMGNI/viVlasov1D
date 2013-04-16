'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.Toolbox import Toolbox


cdef class PETScVlasovSolver(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DA da1, DA da2, DA dax, Vec H0,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 coll_freq=0.0):
        '''
        Constructor
        '''
        
        # distributed array
        self.da1 = da1
        
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
        
        self.nu = coll_freq
        
        # velocity grid
        self.v = v.copy()
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # create history vectors
        self.Fh  = self.da1.createGlobalVec()
        self.H1  = self.da1.createGlobalVec()
        self.H1h = self.da1.createGlobalVec()
        
        # create local vectors
        self.localB   = da1.createLocalVec()
        self.localF   = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        
        self.localH0  = da1.createLocalVec()
        self.localH1  = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
        
    
    def update_potential(self, Vec H1):
        H1.copy(self.H1)
        
    
    def update_history(self, Vec F, Vec H1):
        F.copy(self.Fh)
        H1.copy(self.H1h)
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
    
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec F, Vec Y):
        cdef np.int64_t i, j
        cdef np.int64_t ix, jx
        cdef np.int64_t xe, xs
        
        (xs, xe), = self.da1.getRanges()
        
        self.da1.globalToLocal(F,        self.localF)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.H1,  self.localH1)
        self.da1.globalToLocal(self.H1h, self.localH1h)
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = self.da1.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] f   = self.da1.getVecArray(self.localF  )[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1  = self.da1.getVecArray(self.localH1 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] h1h = self.da1.getVecArray(self.localH1h)[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(0, self.nv):
            
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = f[ix, j]
                    
                else:
                    y[iy, j] = self.toolbox.time_derivative_J1(f, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J1(f,  h0,  ix, j) \
                             + 0.5 * self.toolbox.arakawa_J1(f,  h1h, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J1(fh, h1,  ix, j)
                    
        
    
    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.int64_t i, j, ix, jx
        cdef np.int64_t xs, xe
        
        (xs, xe), = self.da1.getRanges()
        
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        
        cdef np.ndarray[np.float64_t, ndim=2] b   = self.da1.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(0, self.nv):
                
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[iy, j] = 0.0
                    
                else:
                    b[iy, j] = self.toolbox.time_derivative_J1(fh, ix, j) \
                             - 0.5 * self.toolbox.arakawa_J1(fh, h0,  ix, j)
    
