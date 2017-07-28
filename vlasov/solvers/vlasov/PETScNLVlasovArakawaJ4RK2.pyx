'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from vlasov.toolbox.Toolbox import Toolbox


cdef class PETScVlasovSolver(PETScVlasovSolverBase):
    '''
    Implements a variational integrator with fourth order
    symplectic Runge-Kutta time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 object da1  not None,
                 Grid grid not None,
                 Vec H0  not None,
                 Vec H1p not None,
                 Vec H1h not None,
                 Vec H2p not None,
                 Vec H2h not None,
                 Vec H11 not None,
                 Vec H21 not None,
                 double charge=-1.,
                 double coll_freq=0.,
                 double coll_diff=1.,
                 double coll_drag=1.,
                 double regularisation=0.):
        '''
        Constructor
        '''
        
        # initialise parent class
        super().__init__(da1, grid, H0, H1p, H1h, H2p, H2h, charge, coll_freq, coll_diff, coll_drag, regularisation)
        
        # create global data arrays
        self.H11 = H11
        self.H21 = H21
        
        # create local data arrays
        self.localK   = self.da1.createLocalVec()

        self.localH11 = self.da1.createLocalVec()
        self.localH21 = self.da1.createLocalVec()

        
    cpdef update_previous2(self):
        self.H0.copy(self.Have)
        self.Have.axpy(1., self.H11)
        self.Have.axpy(1., self.H21)
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef jacobian(self, Vec K, Vec Y):
        cdef int a, i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        cdef double jpp_J1, jpc_J1, jcp_J1
        cdef double jcc_J2, jpc_J2, jcp_J2
        cdef double result_J1, result_J2, result_J4, poisson
        
        
        K.copy(self.Fave)
        self.Fave.scale(0.5 * self.grid.ht)
        
        cdef double[:,:] h = getLocalArray(self.da1, self.Have, self.localHave)
        cdef double[:,:] f = getLocalArray(self.da1, self.Fave, self.localFave)
        cdef double[:,:] k = getLocalArray(self.da1, K, self.localK)
        cdef double[:,:] y = getGlobalArray(self.da1, Y)
        
        
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
    cpdef function(self, Vec K, Vec Y):
        cdef int a, i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        self.Fh.copy(self.Fave)
        self.Fave.axpy(0.5 * self.grid.ht, K)
        
        cdef double[:,:] h = getLocalArray(self.da1, self.Have, self.localHave)
        cdef double[:,:] f = getLocalArray(self.da1, self.Fave, self.localFave)
        cdef double[:,:] k = getLocalArray(self.da1, K, self.localK)
        cdef double[:,:] y = getGlobalArray(self.da1, Y)
        
        
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
