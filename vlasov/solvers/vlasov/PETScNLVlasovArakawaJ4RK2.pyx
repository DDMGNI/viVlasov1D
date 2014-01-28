'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as npy
cimport numpy as npy

from petsc4py import PETSc

from vlasov.toolbox.Toolbox import Toolbox


cdef class PETScVlasovSolver(PETScVlasovSolverBase):
    '''
    Implements a variational integrator with fourth order
    symplectic Runge-Kutta time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 VIDA da1  not None,
                 Grid grid not None,
                 Vec H0  not None,
                 Vec H1p not None,
                 Vec H1h not None,
                 Vec H2p not None,
                 Vec H2h not None,
                 npy.float64_t charge=-1.,
                 npy.float64_t coll_freq=0.,
                 npy.float64_t coll_diff=1.,
                 npy.float64_t coll_drag=1.,
                 npy.float64_t regularisation=0.):
        '''
        Constructor
        '''
        
        # initialise parent class
        super().__init__(da1, grid, H0, H1p, H1h, H2p, H2h, charge, coll_freq, coll_diff, coll_drag, regularisation)
        
        # create global data arrays
        self.F1  = self.da1.createGlobalVec()
        
        self.H11 = self.da1.createGlobalVec()
        self.H21 = self.da1.createGlobalVec()
        
        # create local data arrays
        self.localK1  = self.da1.createLocalVec()
        self.localF1  = self.da1.createLocalVec()

        self.localH11 = self.da1.createLocalVec()
        self.localH21 = self.da1.createLocalVec()

        
    
    cpdef update_previous2(self, Vec F1, Vec P1int, Vec P1ext):
        F1.copy(self.F1)
        
        self.toolbox.potential_to_hamiltonian(P1int, self.H11)
        self.toolbox.potential_to_hamiltonian(P1ext, self.H21)
        
        self.H0.copy(self.Have)
        self.Have.axpy(1., self.H11)
        self.Have.axpy(1., self.H21)
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(self, Vec K1, Vec Y):
        cdef npy.int64_t a, i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef double jpp_J1, jpc_J1, jcp_J1
        cdef double jcc_J2, jpc_J2, jcp_J2
        cdef double result_J1, result_J2, result_J4, poisson
        
        
        K1.copy(self.F1)
        self.F1.scale(0.5 * self.grid.ht)
        
        cdef double[:,:] h = self.da1.getLocalArray(self.Have, self.localHave)
        cdef double[:,:] f = self.da1.getLocalArray(self.F1, self.localF1)
        cdef double[:,:] k = self.da1.getLocalArray(K1, self.localK1)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy] = k[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    # Araaawa's J1
                    jpp_J1 = (f[ix+1, jx  ] - f[ix-1, jx  ]) * (h[ix,   jx+1] - h[ix,   jx-1]) \
                           - (f[ix,   jx+1] - f[ix,   jx-1]) * (h[ix+1, jx  ] - h[ix-1, jx  ])
                    
                    jpc_J1 = f[ix+1, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - f[ix-1, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - f[ix,   jx+1] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + f[ix,   jx-1] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J1 = f[ix+1, jx+1] * (h[ix,   jx+1] - h[ix+1, jx  ]) \
                           - f[ix-1, jx-1] * (h[ix-1, jx  ] - h[ix,   jx-1]) \
                           - f[ix-1, jx+1] * (h[ix,   jx+1] - h[ix-1, jx  ]) \
                           + f[ix+1, jx-1] * (h[ix+1, jx  ] - h[ix,   jx-1])
                    
                    # Araaawa's J2
                    jcc_J2 = (f[ix+1, jx+1] - f[ix-1, jx-1]) * (h[ix-1, jx+1] - h[ix+1, jx-1]) \
                           - (f[ix-1, jx+1] - f[ix+1, jx-1]) * (h[ix+1, jx+1] - h[ix-1, jx-1])
                    
                    jpc_J2 = f[ix+2, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - f[ix-2, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - f[ix,   jx+2] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + f[ix,   jx-2] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J2 = f[ix+1, jx+1] * (h[ix,   jx+2] - h[ix+2, jx  ]) \
                           - f[ix-1, jx-1] * (h[ix-2, jx  ] - h[ix,   jx-2]) \
                           - f[ix-1, jx+1] * (h[ix,   jx+2] - h[ix-2, jx  ]) \
                           + f[ix+1, jx-1] * (h[ix+2, jx  ] - h[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    poisson   = result_J4 * self.grid.hx_inv * self.grid.hv_inv
                    
                    # solution
                    y[iy, jy] = k[ix, jx] + poisson

    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec K1, Vec Y):
        cdef npy.int64_t a, i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        self.Fh.copy(self.Fave)
        self.Fave.axpy(0.5 * self.grid.ht, K1)
        
        cdef double[:,:] h = self.da1.getLocalArray(self.Have, self.localHave)
        cdef double[:,:] f = self.da1.getLocalArray(self.Fave, self.localFave)
        cdef double[:,:] k = self.da1.getLocalArray(K1, self.localK1)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy] = k[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    # Araaawa's J1
                    jpp_J1 = (f[ix+1, jx  ] - f[ix-1, jx  ]) * (h[ix,   jx+1] - h[ix,   jx-1]) \
                           - (f[ix,   jx+1] - f[ix,   jx-1]) * (h[ix+1, jx  ] - h[ix-1, jx  ])
                    
                    jpc_J1 = f[ix+1, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - f[ix-1, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - f[ix,   jx+1] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + f[ix,   jx-1] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J1 = f[ix+1, jx+1] * (h[ix,   jx+1] - h[ix+1, jx  ]) \
                           - f[ix-1, jx-1] * (h[ix-1, jx  ] - h[ix,   jx-1]) \
                           - f[ix-1, jx+1] * (h[ix,   jx+1] - h[ix-1, jx  ]) \
                           + f[ix+1, jx-1] * (h[ix+1, jx  ] - h[ix,   jx-1])
                    
                    # Araaawa's J2
                    jcc_J2 = (f[ix+1, jx+1] - f[ix-1, jx-1]) * (h[ix-1, jx+1] - h[ix+1, jx-1]) \
                           - (f[ix-1, jx+1] - f[ix+1, jx-1]) * (h[ix+1, jx+1] - h[ix-1, jx-1])
                    
                    jpc_J2 = f[ix+2, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - f[ix-2, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - f[ix,   jx+2] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + f[ix,   jx-2] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J2 = f[ix+1, jx+1] * (h[ix,   jx+2] - h[ix+2, jx  ]) \
                           - f[ix-1, jx-1] * (h[ix-2, jx  ] - h[ix,   jx-2]) \
                           - f[ix-1, jx+1] * (h[ix,   jx+2] - h[ix-2, jx  ]) \
                           + f[ix+1, jx-1] * (h[ix+2, jx  ] - h[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    poisson   = result_J4 * self.grid.hx_inv * self.grid.hv_inv
                    
                    # solution
                    y[iy, jy] = k[ix, jx] + poisson
