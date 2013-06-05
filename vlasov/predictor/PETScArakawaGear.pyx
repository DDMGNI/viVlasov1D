'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.Toolbox import Toolbox


cdef class PETScArakawaGear(object):
    '''
    PETSc/Cython Implementation of Explicit Arakawa-RK4 Vlasov-Poisson Solver
    '''
    
    
    def __init__(self, VIDA da1, VIDA da2, VIDA dax, Vec H0,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv):
        '''
        Constructor
        '''
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv

        # distributed array
        self.da1 = da1
        
        # velocity grid
        self.v = v.copy()
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # potential Hamiltonian history
        self.H1h1 = self.da1.createGlobalVec()
        self.H1h2 = self.da1.createGlobalVec()
        self.H1h3 = self.da1.createGlobalVec()
        self.H1h4 = self.da1.createGlobalVec()
        
        # distribution function history
        self.Fh1 = self.da1.createGlobalVec()
        self.Fh2 = self.da1.createGlobalVec()
        self.Fh3 = self.da1.createGlobalVec()
        self.Fh4 = self.da1.createGlobalVec()
        
        # create global vectors
        self.X1 = self.da1.createGlobalVec()
        self.X2 = self.da1.createGlobalVec()
        self.X3 = self.da1.createGlobalVec()
        self.X4 = self.da1.createGlobalVec()
        
        # create local vectors
        self.localH0   = da1.createLocalVec()
        
        self.localH1h1 = da1.createLocalVec()
        self.localH1h2 = da1.createLocalVec()
        self.localH1h3 = da1.createLocalVec()
        self.localH1h4 = da1.createLocalVec()
        
        self.localFh1  = da1.createLocalVec()
        self.localFh2  = da1.createLocalVec()
        self.localFh3  = da1.createLocalVec()
        self.localFh4  = da1.createLocalVec()
        
        # create toolbox object
        self.toolbox = Toolbox(da1, da2, dax, v, nx, nv, ht, hx, hv)
    
    
    def update_history(self, Vec F, Vec H1):
        self.Fh3.copy(self.Fh4)
        self.Fh2.copy(self.Fh3)
        self.Fh1.copy(self.Fh2)
        
        F.copy(self.Fh1)
        
        self.H1h3.copy(self.H1h4)
        self.H1h2.copy(self.H1h3)
        self.H1h1.copy(self.H1h2)
        
        H1.copy(self.H1h1)
        
    
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
        
        
        (xs, xe), = self.da1.getRanges()
        
        for i in np.arange(xs, xe):
            for j in np.arange(0, self.nv):
                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = 2./3. * (
                                         + 2.  * fh1[ix,j]
                                         - 0.5 * fh2[ix,j]
                                         - 2.  * self.ht * self.toolbox.arakawa_J4(fh1, hh1, ix, j)
                                         + 1.  * self.ht * self.toolbox.arakawa_J4(fh2, hh2, ix, j)
                                       )
    
    
    def gear3(self, Vec Y):
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] y    = self.da1.getGlobalArray(Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] h0   = self.da1.getLocalArray(self.H0,   self.localH0  )
        cdef np.ndarray[np.float64_t, ndim=2] h1h1 = self.da1.getLocalArray(self.H1h1, self.localH1h1)
        cdef np.ndarray[np.float64_t, ndim=2] h1h2 = self.da1.getLocalArray(self.H1h2, self.localH1h2)
        cdef np.ndarray[np.float64_t, ndim=2] h1h3 = self.da1.getLocalArray(self.H1h2, self.localH1h3)
        cdef np.ndarray[np.float64_t, ndim=2] fh1  = self.da1.getLocalArray(self.Fh1,  self.localFh1 )
        cdef np.ndarray[np.float64_t, ndim=2] fh2  = self.da1.getLocalArray(self.Fh2,  self.localFh2 )
        cdef np.ndarray[np.float64_t, ndim=2] fh3  = self.da1.getLocalArray(self.Fh2,  self.localFh3 )
        
        cdef np.ndarray[np.float64_t, ndim=2] hh1 = h0 + h1h1
        cdef np.ndarray[np.float64_t, ndim=2] hh2 = h0 + h1h2
        cdef np.ndarray[np.float64_t, ndim=2] hh3 = h0 + h1h3
        
        
        (xs, xe), = self.da1.getRanges()
        
        for i in np.arange(xs, xe):
            for j in np.arange(0, self.nv):
                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = 6./11. * (
                                         + 3.   * fh1[ix,j]
                                         - 1.5  * fh2[ix,j]
                                         + 1./3.* fh3[ix,j]
                                         - 3.   * self.ht * self.toolbox.arakawa_J4(fh1, hh1, ix, j)
                                         + 3.   * self.ht * self.toolbox.arakawa_J4(fh2, hh2, ix, j)
                                         - 1.   * self.ht * self.toolbox.arakawa_J4(fh3, hh3, ix, j)
                                       )


    def gear4(self, Vec Y):
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] y    = self.da1.getGlobalArray(Y)
        
        cdef np.ndarray[np.float64_t, ndim=2] h0   = self.da1.getLocalArray(self.H0,   self.localH0  )
        cdef np.ndarray[np.float64_t, ndim=2] h1h1 = self.da1.getLocalArray(self.H1h1, self.localH1h1)
        cdef np.ndarray[np.float64_t, ndim=2] h1h2 = self.da1.getLocalArray(self.H1h2, self.localH1h2)
        cdef np.ndarray[np.float64_t, ndim=2] h1h3 = self.da1.getLocalArray(self.H1h2, self.localH1h3)
        cdef np.ndarray[np.float64_t, ndim=2] h1h4 = self.da1.getLocalArray(self.H1h2, self.localH1h4)
        cdef np.ndarray[np.float64_t, ndim=2] fh1  = self.da1.getLocalArray(self.Fh1,  self.localFh1 )
        cdef np.ndarray[np.float64_t, ndim=2] fh2  = self.da1.getLocalArray(self.Fh2,  self.localFh2 )
        cdef np.ndarray[np.float64_t, ndim=2] fh3  = self.da1.getLocalArray(self.Fh2,  self.localFh3 )
        cdef np.ndarray[np.float64_t, ndim=2] fh4  = self.da1.getLocalArray(self.Fh2,  self.localFh4 )
        
        cdef np.ndarray[np.float64_t, ndim=2] hh1 = h0 + h1h1
        cdef np.ndarray[np.float64_t, ndim=2] hh2 = h0 + h1h2
        cdef np.ndarray[np.float64_t, ndim=2] hh3 = h0 + h1h3
        cdef np.ndarray[np.float64_t, ndim=2] hh4 = h0 + h1h4
        
        
        (xs, xe), = self.da1.getRanges()
        
        for i in np.arange(xs, xe):
            for j in np.arange(0, self.nv):
                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = 12./25. * (
                                         + 4.   * fh1[ix,j]
                                         - 3.   * fh2[ix,j]
                                         + 4./3.* fh3[ix,j]
                                         - 0.25 * fh4[ix,j]
                                         - 4.   * self.ht * self.toolbox.arakawa_J4(fh1, hh1, ix, j)
                                         + 6.   * self.ht * self.toolbox.arakawa_J4(fh2, hh2, ix, j)
                                         - 4.   * self.ht * self.toolbox.arakawa_J4(fh3, hh3, ix, j)
                                         + 1.   * self.ht * self.toolbox.arakawa_J4(fh4, hh4, ix, j)
                                       )
