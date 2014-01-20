'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.toolbox.Arakawa import Arakawa


cdef class PETScArakawaGear(PETScExplicitSolver):
    '''
    PETSc/Cython Implementation of Explicit Arakawa-Gear Vlasov-Poisson Solver
    '''
    
    
    def __init__(self, 
                 VIDA da1  not None,
                 Grid grid not None,
                 Vec H0    not None,
                 Vec H1    not None,
                 Vec H2    not None):
        '''
        Constructor
        '''
        
        super().__init__(da1, grid, H0, H1, H2)
        
        # potential Hamiltonian history
        self.H1h1 = self.da1.createGlobalVec()
        self.H1h2 = self.da1.createGlobalVec()
        self.H1h3 = self.da1.createGlobalVec()
        self.H1h4 = self.da1.createGlobalVec()
        
        # external Hamiltonian history
        self.H2h1 = self.da1.createGlobalVec()
        self.H2h2 = self.da1.createGlobalVec()
        self.H2h3 = self.da1.createGlobalVec()
        self.H2h4 = self.da1.createGlobalVec()
        
        # distribution function history
        self.Fh1 = self.da1.createGlobalVec()
        self.Fh2 = self.da1.createGlobalVec()
        self.Fh3 = self.da1.createGlobalVec()
        self.Fh4 = self.da1.createGlobalVec()
        
        # create local vectors
        self.localH1h1 = da1.createLocalVec()
        self.localH1h2 = da1.createLocalVec()
        self.localH1h3 = da1.createLocalVec()
        self.localH1h4 = da1.createLocalVec()
        
        self.localH2h1 = da1.createLocalVec()
        self.localH2h2 = da1.createLocalVec()
        self.localH2h3 = da1.createLocalVec()
        self.localH2h4 = da1.createLocalVec()
        
        self.localFh1  = da1.createLocalVec()
        self.localFh2  = da1.createLocalVec()
        self.localFh3  = da1.createLocalVec()
        self.localFh4  = da1.createLocalVec()
        
    
    def update_history(self, Vec F):
        self.Fh3.copy(self.Fh4)
        self.Fh2.copy(self.Fh3)
        self.Fh1.copy(self.Fh2)
        
        F.copy(self.Fh1)
        
        
        self.H1h3.copy(self.H1h4)
        self.H1h2.copy(self.H1h3)
        self.H1h1.copy(self.H1h2)
        
        self.H1.copy(self.H1h1)
        
        
        self.H2h3.copy(self.H2h4)
        self.H2h2.copy(self.H2h3)
        self.H2h1.copy(self.H2h2)
        
        self.H2.copy(self.H2h1)
        
    
    def gear2(self, Vec Y):
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] y    = self.da1.getGlobalArray(Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] h0   = self.da1.getLocalArray(self.H0,   self.localH0  )
        cdef np.ndarray[np.float64_t, ndim=2] h1h1 = self.da1.getLocalArray(self.H1h1, self.localH1h1)
        cdef np.ndarray[np.float64_t, ndim=2] h1h2 = self.da1.getLocalArray(self.H1h2, self.localH1h2)
        cdef np.ndarray[np.float64_t, ndim=2] fh1  = self.da1.getLocalArray(self.Fh1,  self.localFh1 )
        cdef np.ndarray[np.float64_t, ndim=2] fh2  = self.da1.getLocalArray(self.Fh2,  self.localFh2 )
        
        cdef np.ndarray[np.float64_t, ndim=2] hh1 = h0 + h1h1
        cdef np.ndarray[np.float64_t, ndim=2] hh2 = h0 + h1h2
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.grid.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, jy] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, jy] = 2./3. * (
                                         + 2.  * fh1[ix, jx]
                                         - 0.5 * fh2[ix, jx]
                                         - 2.  * self.grid.ht * self.arakawa.arakawa_J4(fh1, hh1, ix, j)
                                         + 1.  * self.grid.ht * self.arakawa.arakawa_J4(fh2, hh2, ix, j)
                                       )
    
    
    def gear3(self, Vec Y):
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] y    = self.da1.getGlobalArray(Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] h0   = self.da1.getLocalArray(self.H0,   self.localH0  )
        cdef np.ndarray[np.float64_t, ndim=2] h1h1 = self.da1.getLocalArray(self.H1h1, self.localH1h1)
        cdef np.ndarray[np.float64_t, ndim=2] h1h2 = self.da1.getLocalArray(self.H1h2, self.localH1h2)
        cdef np.ndarray[np.float64_t, ndim=2] h1h3 = self.da1.getLocalArray(self.H1h3, self.localH1h3)
        cdef np.ndarray[np.float64_t, ndim=2] fh1  = self.da1.getLocalArray(self.Fh1,  self.localFh1 )
        cdef np.ndarray[np.float64_t, ndim=2] fh2  = self.da1.getLocalArray(self.Fh2,  self.localFh2 )
        cdef np.ndarray[np.float64_t, ndim=2] fh3  = self.da1.getLocalArray(self.Fh3,  self.localFh3 )
        
        cdef np.ndarray[np.float64_t, ndim=2] hh1 = h0 + h1h1
        cdef np.ndarray[np.float64_t, ndim=2] hh2 = h0 + h1h2
        cdef np.ndarray[np.float64_t, ndim=2] hh3 = h0 + h1h3
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.grid.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, jy] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, jy] = 6./11. * (
                                         + 3.   * fh1[ix, jx]
                                         - 1.5  * fh2[ix, jx]
                                         + 1./3.* fh3[ix, jx]
                                         - 3.   * self.grid.ht * self.arakawa.arakawa_J4(fh1, hh1, ix, j)
                                         + 3.   * self.grid.ht * self.arakawa.arakawa_J4(fh2, hh2, ix, j)
                                         - 1.   * self.grid.ht * self.arakawa.arakawa_J4(fh3, hh3, ix, j)
                                       )


    def gear4(self, Vec Y):
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] y    = self.da1.getGlobalArray(Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] h0   = self.da1.getLocalArray(self.H0,   self.localH0  )
        cdef np.ndarray[np.float64_t, ndim=2] h1h1 = self.da1.getLocalArray(self.H1h1, self.localH1h1)
        cdef np.ndarray[np.float64_t, ndim=2] h1h2 = self.da1.getLocalArray(self.H1h2, self.localH1h2)
        cdef np.ndarray[np.float64_t, ndim=2] h1h3 = self.da1.getLocalArray(self.H1h3, self.localH1h3)
        cdef np.ndarray[np.float64_t, ndim=2] h1h4 = self.da1.getLocalArray(self.H1h4, self.localH1h4)
        cdef np.ndarray[np.float64_t, ndim=2] fh1  = self.da1.getLocalArray(self.Fh1,  self.localFh1 )
        cdef np.ndarray[np.float64_t, ndim=2] fh2  = self.da1.getLocalArray(self.Fh2,  self.localFh2 )
        cdef np.ndarray[np.float64_t, ndim=2] fh3  = self.da1.getLocalArray(self.Fh3,  self.localFh3 )
        cdef np.ndarray[np.float64_t, ndim=2] fh4  = self.da1.getLocalArray(self.Fh4,  self.localFh4 )
        
        cdef np.ndarray[np.float64_t, ndim=2] hh1 = h0 + h1h1
        cdef np.ndarray[np.float64_t, ndim=2] hh2 = h0 + h1h2
        cdef np.ndarray[np.float64_t, ndim=2] hh3 = h0 + h1h3
        cdef np.ndarray[np.float64_t, ndim=2] hh4 = h0 + h1h4
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.grid.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, jy] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, jy] = 12./25. * (
                                         + 4.   * fh1[ix, jx]
                                         - 3.   * fh2[ix, jx]
                                         + 4./3.* fh3[ix, jx]
                                         - 0.25 * fh4[ix, jx]
                                         - 4.   * self.grid.ht * self.arakawa.arakawa_J4(fh1, hh1, ix, j)
                                         + 6.   * self.grid.ht * self.arakawa.arakawa_J4(fh2, hh2, ix, j)
                                         - 4.   * self.grid.ht * self.arakawa.arakawa_J4(fh3, hh3, ix, j)
                                         + 1.   * self.grid.ht * self.arakawa.arakawa_J4(fh4, hh4, ix, j)
                                       )
