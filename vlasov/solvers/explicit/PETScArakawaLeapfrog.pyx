'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Vec


cdef class PETScArakawaLeapfrog(PETScExplicitSolver):
    '''
    PETSc/Cython Implementation of Explicit Arakawa-Leapfrog Vlasov-Poisson Solver
    '''
    
    
    def __init__(self, 
                 config    not None,
                 object da1  not None,
                 Grid grid not None,
                 Vec H0    not None,
                 Vec H1    not None,
                 Vec H2    not None,
                 niter=1):
        '''
        Constructor
        '''
        
        super().__init__(config, da1, grid, H0, H1, H2, niter)
        
        # distribution function history
        self.Fh1 = self.da1.createGlobalVec()
        self.Fh2 = self.da1.createGlobalVec()
        
        # create local vectors
        self.localFh1 = da1.createLocalVec()
        self.localFh2 = da1.createLocalVec()
        
    
    def update_history(self, Vec F, Vec H1):
        self.Fh1.copy(self.Fh2)
        F.copy(self.Fh1)
        
    
    def leapfrog2(self, Vec Y):
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = getGlobalArray(self.da1, Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] h0  = getLocalArray(self.da1, self.H0,  self.localH0 )
        cdef np.ndarray[np.float64_t, ndim=2] h1h = getLocalArray(self.da1, self.H1,  self.localH1 )
        cdef np.ndarray[np.float64_t, ndim=2] h2h = getLocalArray(self.da1, self.H2,  self.localH2 )
        cdef np.ndarray[np.float64_t, ndim=2] fh1 = getLocalArray(self.da1, self.Fh1, self.localFh1)
        cdef np.ndarray[np.float64_t, ndim=2] fh2 = getLocalArray(self.da1, self.Fh2, self.localFh2)
        
        cdef np.ndarray[np.float64_t, ndim=2] hh  = h0 + h1h + h2h
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
                
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    # Dirichlet boundary conditions
                    y[iy, jy] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, jy] = fh2[ix, jx] - 2. * self.grid.ht / float(self.niter) * self.arakawa.poisson_bracket_point(fh1, hh, ix, j)
    
    
    def leapfrog4(self, Vec Y):
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] y   = getGlobalArray(self.da1, Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] h0  = getLocalArray(self.da1, self.H0,  self.localH0 )
        cdef np.ndarray[np.float64_t, ndim=2] h1h = getLocalArray(self.da1, self.H1,  self.localH1 )
        cdef np.ndarray[np.float64_t, ndim=2] h2h = getLocalArray(self.da1, self.H2,  self.localH2 )
        cdef np.ndarray[np.float64_t, ndim=2] fh1 = getLocalArray(self.da1, self.Fh1, self.localFh1)
        cdef np.ndarray[np.float64_t, ndim=2] fh2 = getLocalArray(self.da1, self.Fh2, self.localFh2)
        
        cdef np.ndarray[np.float64_t, ndim=2] hh  = h0 + h1h + h2h
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
                
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    # Dirichlet boundary conditions
                    y[iy, jy] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, jy] = fh2[ix, jx] - 2. * self.grid.ht / float(self.niter) * self.arakawa.poisson_bracket_point(fh1, hh, ix, j)
    
    
