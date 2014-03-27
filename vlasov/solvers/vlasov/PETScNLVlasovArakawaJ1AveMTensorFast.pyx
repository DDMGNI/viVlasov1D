'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np


cdef class PETScVlasovSolver(vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4TensorFast.PETScVlasovSolver):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    cdef formBandedPreconditionerMatrix(self, dcomplex[:,:] matrix, dcomplex eigen):
        cdef int j
        
        cdef double[:] v = self.grid.v
        
        cdef double arak_fac_J1 = 0.5 / (12. * self.grid.hx * self.grid.hv)
        
        
        cdef dcomplex[:] diagm = np.zeros(self.grid.nv, dtype=np.cdouble)
        cdef dcomplex[:] diag  = np.ones (self.grid.nv, dtype=np.cdouble)
        cdef dcomplex[:] diagp = np.zeros(self.grid.nv, dtype=np.cdouble)
        
        for j in range(2, self.grid.nv-2):
            diagm[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] - self.grid.hv2 ) * arak_fac_J1 + 0.25 * self.grid.ht_inv
            diag [j] = eigen *         4. * self.grid.hv * v[j]                   * arak_fac_J1 + 0.50 * self.grid.ht_inv
            diagp[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] + self.grid.hv2 ) * arak_fac_J1 + 0.25 * self.grid.ht_inv
        
        matrix[1, 1:  ] = diagp[:-1]
        matrix[2,  :  ] = diag [:]
        matrix[3,  :-1] = diagm[1:]
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef jacobianSolver(self, Vec F, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        cdef double jpp_J1, jpc_J1, jcp_J1
        cdef double jcc_J2, jpc_J2, jcp_J2
        cdef double result_J1, poisson, time_deriv
        cdef double coll_drag, coll_diff
        cdef double collisions     = 0.
        cdef double regularisation = 0.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double[:,:] fd    = self.da1.getLocalArray(F, self.localFd)
        cdef double[:,:] y     = self.da1.getGlobalArray(Y)
        cdef double[:,:] h_ave = self.da1.getLocalArray(self.Have, self.localHave)
        
        cdef double[:] v = self.grid.v
        cdef double[:] u = self.Up.getArray()
        cdef double[:] a = self.Ap.getArray()
        
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy] = fd[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    # time derivative
                    time_deriv = ( 1. * fd[ix-1, jx-1] + 2. * fd[ix, jx-1] + 1. * fd[ix+1, jx-1]
                                 + 2. * fd[ix-1, jx  ] + 4. * fd[ix, jx  ] + 2. * fd[ix+1, jx  ]
                                 + 1. * fd[ix-1, jx+1] + 2. * fd[ix, jx+1] + 1. * fd[ix+1, jx+1] ) * self.grid.ht_inv / 16.
                    
                    
                    # Arakawa's J1
                    jpp_J1 = (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                           - (fd[ix,   jx+1] - fd[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                    
                    jpc_J1 = fd[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - fd[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - fd[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + fd[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J1 = fd[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                           - fd[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                           - fd[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                           + fd[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    poisson   = 0.5 * result_J1 * self.grid.hx_inv * self.grid.hv_inv
                    
                    
                    # collision operator
                    if self.nu > 0.:
                        coll_drag = ( (v[j+1] - u[ix]) * fd[ix, jx+1] - (v[j-1] - u[ix]) * fd[ix, jx-1] ) * a[ix]
                        
                        coll_diff = ( fd[ix, jx+1] - 2. * fd[ix, jx] + fd[ix, jx-1] )
                        
                        collisions = \
                              - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                              - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv
                    
                    # regularisation
                    if self.regularisation != 0.:
                        regularisation = \
                              + self.grid.ht * self.regularisation * self.grid.hx2_inv * ( 2. * fd[ix, jx] - fd[ix+1, jx] - fd[ix-1, jx] ) \
                              + self.grid.ht * self.regularisation * self.grid.hv2_inv * ( 2. * fd[ix, jx] - fd[ix, jx+1] - fd[ix, jx-1] )
                    
                    # solution
                    y[iy, jy] = time_deriv \
                              + poisson \
                              + collisions \
                              + regularisation
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef functionSolver(self, Vec F, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        cdef double jpp_J1, jpc_J1, jcp_J1
        cdef double jcc_J2, jpc_J2, jcp_J2
        cdef double result_J1, poisson, time_deriv_p, time_deriv_h
        cdef double coll_drag, coll_diff
        cdef double collisions     = 0.
        cdef double regularisation = 0.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.Fave.set(0.)
        self.Fave.axpy(0.5, self.Fh)
        self.Fave.axpy(0.5, F)
        
        cdef double[:,:] y     = self.da1.getGlobalArray(Y)
        cdef double[:,:] fp    = self.da1.getLocalArray(F, self.localFp)
        cdef double[:,:] fh    = self.da1.getLocalArray(self.Fh, self.localFh)
        cdef double[:,:] f_ave = self.da1.getLocalArray(self.Fave, self.localFave)
        cdef double[:,:] h_ave = self.da1.getLocalArray(self.Have, self.localHave)
        
        cdef double[:] v  = self.grid.v
        cdef double[:] up = self.Up.getArray()
        cdef double[:] uh = self.Uh.getArray()
        cdef double[:] ap = self.Ap.getArray()
        cdef double[:] ah = self.Ah.getArray()

        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy] = fp[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
                    
                    # time derivative
                    time_deriv_p = ( 1. * fp[ix-1, jx-1] + 2. * fp[ix, jx-1] + 1. * fp[ix+1, jx-1]
                                   + 2. * fp[ix-1, jx  ] + 4. * fp[ix, jx  ] + 2. * fp[ix+1, jx  ]
                                   + 1. * fp[ix-1, jx+1] + 2. * fp[ix, jx+1] + 1. * fp[ix+1, jx+1] ) * self.grid.ht_inv / 16.
                    
                    time_deriv_h = ( 1. * fh[ix-1, jx-1] + 2. * fh[ix, jx-1] + 1. * fh[ix+1, jx-1]
                                   + 2. * fh[ix-1, jx  ] + 4. * fh[ix, jx  ] + 2. * fh[ix+1, jx  ]
                                   + 1. * fh[ix-1, jx+1] + 2. * fh[ix, jx+1] + 1. * fh[ix+1, jx+1] ) * self.grid.ht_inv / 16.
                    
                    
                    # Arakawa's J1
                    jpp_J1 = (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                           - (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                    
                    jpc_J1 = f_ave[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - f_ave[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - f_ave[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + f_ave[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J1 = f_ave[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                           - f_ave[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                           - f_ave[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                           + f_ave[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    poisson   = result_J1 * self.grid.hx_inv * self.grid.hv_inv
                    
                    
                    # collision operator
                    if self.nu > 0.:
                        coll_drag = ( (v[j+1] - up[ix]) * fp[ix, jx+1] - (v[j-1] - up[ix]) * fp[ix, jx-1] ) * ap[ix] \
                                  + ( (v[j+1] - uh[ix]) * fh[ix, jx+1] - (v[j-1] - uh[ix]) * fh[ix, jx-1] ) * ah[ix]
                        
                        coll_diff = ( fp[ix, jx+1] - 2. * fp[ix, jx] + fp[ix, jx-1] ) \
                                  + ( fh[ix, jx+1] - 2. * fh[ix, jx] + fh[ix, jx-1] )
                        
                        collisions = \
                             - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                             - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv
                    
                    
                    y[iy, jy] = time_deriv_p - time_deriv_h \
                             + poisson \
                             + collisions
