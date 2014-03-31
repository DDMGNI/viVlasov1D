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
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 VIDA da2  not None,
                 VIDA da1  not None,
                 Grid grid not None,
                 Vec H0  not None,
                 Vec H1p not None,
                 Vec H1h not None,
                 Vec H2p not None,
                 Vec H2h not None,
                 double charge=-1.,
                 double coll_freq=1.):
        
        # initialise parent class
        super().__init__(da1, grid, H0, H1p, H1h, H2p, H2h, charge, coll_freq, 0., 0., 0.)
        
        # distributed array
        self.da2 = da2
        
        # create global vectors
        self.Ft   = self.da1.createGlobalVec()
        self.Gt   = self.da1.createGlobalVec()
        self.Gp   = self.da1.createGlobalVec()
        self.Gh   = self.da1.createGlobalVec()
        self.Gave = self.da1.createGlobalVec()
        
        # create local vectors
        self.localK    = self.da2.createLocalVec()
        self.localGp   = self.da1.createLocalVec()
        self.localGh   = self.da1.createLocalVec()
        self.localGd   = self.da1.createLocalVec()
        self.localGave = self.da1.createLocalVec()
    
    
    
    def update_history_db(self, Vec F, Vec G):
        self.update_history(F)
        G.copy(self.Gh)
    
    def update_previous_db(self, Vec F, Vec G):
        self.update_previous(F)
        G.copy(self.Gp)
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(self, Vec K, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        cdef double jpp_J1_f, jpc_J1_f, jcp_J1_f
        cdef double jpp_J1_g, jpc_J1_g, jcp_J1_g
        cdef double jpp_J1_h, jpc_J1_h, jcp_J1_h
        cdef double result_J1_f, result_J1_g, result_J1_h
        cdef double poisson, double_bracket
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.Fave.set(0.)
        self.Fave.axpy(0.5, self.Fh)
        self.Fave.axpy(0.5, self.Fp)
        
        self.Gave.set(0.)
        self.Gave.axpy(0.5, self.Gh)
        self.Gave.axpy(0.5, self.Gp)
        
        cdef double[:,:,:] k     = self.da2.getGlobalArray(K)
        cdef double[:,:,:] y     = self.da2.getGlobalArray(Y)
        
        cdef np.ndarray[dtype=np.float64_t, ndim=2] ft = self.da1.getGlobalArray(self.Ft)
        cdef np.ndarray[dtype=np.float64_t, ndim=2] gt = self.da1.getGlobalArray(self.Gt)
        
        ft[...] = k[:,:,0]
        gt[...] = k[:,:,1]
        
        cdef double[:,:]   fd    = self.da1.getLocalArray(self.Ft, self.localFd)
        cdef double[:,:]   gd    = self.da1.getLocalArray(self.Gt, self.localGd)
        cdef double[:,:]   f_ave = self.da1.getLocalArray(self.Fave, self.localFave)
        cdef double[:,:]   g_ave = self.da1.getLocalArray(self.Gave, self.localGave)
        cdef double[:,:]   h_ave = self.da1.getLocalArray(self.Have, self.localHave)
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy, 0] = fd[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                y[0:xe-xs, jy, 1] = gd[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    # Arakawa's J1
                    jpp_J1_f = (gd[ix+1, jx  ] - gd[ix-1, jx  ]) * (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) \
                             - (gd[ix,   jx+1] - gd[ix,   jx-1]) * (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ])
                    
                    jpc_J1_f = gd[ix+1, jx  ] * (f_ave[ix+1, jx+1] - f_ave[ix+1, jx-1]) \
                             - gd[ix-1, jx  ] * (f_ave[ix-1, jx+1] - f_ave[ix-1, jx-1]) \
                             - gd[ix,   jx+1] * (f_ave[ix+1, jx+1] - f_ave[ix-1, jx+1]) \
                             + gd[ix,   jx-1] * (f_ave[ix+1, jx-1] - f_ave[ix-1, jx-1])
                    
                    jcp_J1_f = gd[ix+1, jx+1] * (f_ave[ix,   jx+1] - f_ave[ix+1, jx  ]) \
                             - gd[ix-1, jx-1] * (f_ave[ix-1, jx  ] - f_ave[ix,   jx-1]) \
                             - gd[ix-1, jx+1] * (f_ave[ix,   jx+1] - f_ave[ix-1, jx  ]) \
                             + gd[ix+1, jx-1] * (f_ave[ix+1, jx  ] - f_ave[ix,   jx-1])
                    
                    jpp_J1_g = (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (g_ave[ix,   jx+1] - g_ave[ix,   jx-1]) \
                             - (fd[ix,   jx+1] - fd[ix,   jx-1]) * (g_ave[ix+1, jx  ] - g_ave[ix-1, jx  ])
                    
                    jpc_J1_g = fd[ix+1, jx  ] * (g_ave[ix+1, jx+1] - g_ave[ix+1, jx-1]) \
                             - fd[ix-1, jx  ] * (g_ave[ix-1, jx+1] - g_ave[ix-1, jx-1]) \
                             - fd[ix,   jx+1] * (g_ave[ix+1, jx+1] - g_ave[ix-1, jx+1]) \
                             + fd[ix,   jx-1] * (g_ave[ix+1, jx-1] - g_ave[ix-1, jx-1])
                    
                    jcp_J1_g = fd[ix+1, jx+1] * (g_ave[ix,   jx+1] - g_ave[ix+1, jx  ]) \
                             - fd[ix-1, jx-1] * (g_ave[ix-1, jx  ] - g_ave[ix,   jx-1]) \
                             - fd[ix-1, jx+1] * (g_ave[ix,   jx+1] - g_ave[ix-1, jx  ]) \
                             + fd[ix+1, jx-1] * (g_ave[ix+1, jx  ] - g_ave[ix,   jx-1])
                    
                    jpp_J1_h = (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                             - (fd[ix,   jx+1] - fd[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                    
                    jpc_J1_h = fd[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                             - fd[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                             - fd[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                             + fd[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J1_h = fd[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                             - fd[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                             - fd[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                             + fd[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                    
                    result_J1_f = (jpp_J1_f + jpc_J1_f + jcp_J1_f) / 12.
                    result_J1_g = (jpp_J1_g + jpc_J1_g + jcp_J1_g) / 12.
                    result_J1_h = (jpp_J1_h + jpc_J1_h + jcp_J1_h) / 12.
                    
                    double_bracket = 0.5 * result_J1_f * self.grid.hx_inv * self.grid.hv_inv \
                                   + 0.5 * result_J1_g * self.grid.hx_inv * self.grid.hv_inv
                    poisson        = 0.5 * result_J1_h * self.grid.hx_inv * self.grid.hv_inv
                    
                    
                    # solution
                    y[iy, jy, 0] = fd[ix, jx] * self.grid.ht_inv \
                                 + poisson \
                                 - self.nu * double_bracket
                    
                    y[iy, jy, 1] = 0.5 * gd[ix, jx] \
                                 - poisson
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec K, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        cdef double jpp_J1_g, jpc_J1_g, jcp_J1_g
        cdef double jpp_J1_h, jpc_J1_h, jcp_J1_h
        cdef double result_J1_g, result_J1_h
        cdef double poisson, double_bracket
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double[:,:,:] k  = self.da2.getGlobalArray(K)
        cdef double[:,:,:] y  = self.da2.getGlobalArray(Y)
        
        cdef np.ndarray[dtype=np.float64_t, ndim=2] ft = self.da1.getGlobalArray(self.Ft)
        cdef np.ndarray[dtype=np.float64_t, ndim=2] gt = self.da1.getGlobalArray(self.Gt)
        
        ft[...] = k[:,:,0]
        gt[...] = k[:,:,1]
        
        self.Fave.set(0.)
        self.Fave.axpy(0.5, self.Fh)
        self.Fave.axpy(0.5, self.Ft)
        
        self.Gave.set(0.)
        self.Gave.axpy(0.5, self.Gh)
        self.Gave.axpy(0.5, self.Gt)
        
        self.Fder.set(0.)
        self.Fder.axpy(+1, self.Ft)
        self.Fder.axpy(-1, self.Fh)
        
        cdef double[:,:] f_ave = self.da1.getLocalArray(self.Fave, self.localFave)
        cdef double[:,:] g_ave = self.da1.getLocalArray(self.Gave, self.localGave)
        cdef double[:,:] h_ave = self.da1.getLocalArray(self.Have, self.localHave)
        cdef double[:,:] f_der = self.da1.getLocalArray(self.Fder, self.localFder)
        
        cdef double[:,:] fp    = self.da1.getLocalArray(self.Ft, self.localFp)
        cdef double[:,:] gp    = self.da1.getLocalArray(self.Gt, self.localGp)
        
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy, 0] = fp[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                y[0:xe-xs, jy, 1] = gp[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
                    
                    # Arakawa's J1
                    jpp_J1_g = (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ]) * (g_ave[ix,   jx+1] - g_ave[ix,   jx-1]) \
                             - (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) * (g_ave[ix+1, jx  ] - g_ave[ix-1, jx  ])
                    
                    jpc_J1_g = f_ave[ix+1, jx  ] * (g_ave[ix+1, jx+1] - g_ave[ix+1, jx-1]) \
                             - f_ave[ix-1, jx  ] * (g_ave[ix-1, jx+1] - g_ave[ix-1, jx-1]) \
                             - f_ave[ix,   jx+1] * (g_ave[ix+1, jx+1] - g_ave[ix-1, jx+1]) \
                             + f_ave[ix,   jx-1] * (g_ave[ix+1, jx-1] - g_ave[ix-1, jx-1])
                    
                    jcp_J1_g = f_ave[ix+1, jx+1] * (g_ave[ix,   jx+1] - g_ave[ix+1, jx  ]) \
                             - f_ave[ix-1, jx-1] * (g_ave[ix-1, jx  ] - g_ave[ix,   jx-1]) \
                             - f_ave[ix-1, jx+1] * (g_ave[ix,   jx+1] - g_ave[ix-1, jx  ]) \
                             + f_ave[ix+1, jx-1] * (g_ave[ix+1, jx  ] - g_ave[ix,   jx-1])
                    
                    jpp_J1_h = (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                             - (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                    
                    jpc_J1_h = f_ave[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                             - f_ave[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                             - f_ave[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                             + f_ave[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J1_h = f_ave[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                             - f_ave[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                             - f_ave[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                             + f_ave[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                    
                    result_J1_g = (jpp_J1_g + jpc_J1_g + jcp_J1_g) / 12.
                    result_J1_h = (jpp_J1_h + jpc_J1_h + jcp_J1_h) / 12.
                    
                    double_bracket = 0.5 * result_J1_g * self.grid.hx_inv * self.grid.hv_inv
                    poisson        = 0.5 * result_J1_h * self.grid.hx_inv * self.grid.hv_inv
                    
                    
                    # solution
                    y[iy, jy, 0] = f_der[ix, jx] * self.grid.ht_inv \
                                 + poisson \
                                 - self.nu * double_bracket
                    
                    y[iy, jy, 1] = g_ave[ix, jx] \
                                 - poisson

