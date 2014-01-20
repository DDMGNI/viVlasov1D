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
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def formJacobian(self, Mat A):
        cdef npy.int64_t i, j, ix
        cdef npy.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.get_data_arrays()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        
#         cdef npy.float64_t time_fac      = 0.
#         cdef npy.float64_t arak_fac_J1   = 0.
#         cdef npy.float64_t arak_fac_J2   = 0.
#         cdef npy.float64_t coll_drag_fac = 0.
#         cdef npy.float64_t coll_diff_fac = 0.
        
        cdef npy.float64_t time_fac      = 1.0  / self.grid.ht
        cdef npy.float64_t arak_fac_J1   = 0.5 / (12. * self.grid.hx * self.grid.hv)
        
        cdef npy.float64_t coll_drag_fac = - 0.5 * self.coll_drag * self.nu * self.grid.hv_inv * 0.5
        cdef npy.float64_t coll_diff_fac = - 0.5 * self.coll_diff * self.nu * self.grid.hv2_inv
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        # Vlasov Equation
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            
            row.index = (i,)
                
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                row.field = j
                
                # Dirichlet boundary conditions
                if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
                    A.setValueStencil(row, row, 1.0)
                    
                else:
                    
                    for index, value in [
                            ((i-1, j-1), - (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) * arak_fac_J1),
                            ((i-1, j  ), - (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) * arak_fac_J1 \
                                         - (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J1),
                            ((i-1, j+1), - (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) * arak_fac_J1),
                            
                            ((i,   j-1), + (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
                                         + (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) * arak_fac_J1 \
                                         - coll_drag_fac * ( self.np[ix  ] * self.v[jx-1] - self.up[ix  ] ) * self.ap[ix  ] \
                                         + coll_diff_fac),
                            ((i,   j  ), + time_fac
                                         - 2. * coll_diff_fac),
                            ((i,   j+1), - (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
                                         - (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) * arak_fac_J1 \
                                         + coll_drag_fac * ( self.np[ix  ] * self.v[jx+1] - self.up[ix  ] ) * self.ap[ix  ] \
                                         + coll_diff_fac),
                            
                            ((i+1, j-1), + (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1]) * arak_fac_J1),
                            ((i+1, j  ), + (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) * arak_fac_J1 \
                                         + (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J1),
                            ((i+1, j+1), + (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) * arak_fac_J1),
                        ]:

                        col.index = index
                        col.field = 0
                        A.setValueStencil(row, col, value)
                        
        
        A.assemble()



    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        cdef npy.float64_t coll_drag, coll_diff
        
        self.get_data_arrays()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fd = self.da1.getLocalArray(F, self.localFd)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v     = self.v
        cdef npy.ndarray[npy.float64_t, ndim=1] u     = self.up
        cdef npy.ndarray[npy.float64_t, ndim=1] a     = self.ap
        
        
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = fd[ix, jx]
                    
                else:
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
         
                    # collision operator
                    coll_drag = ( (v[jx+1] - u[ix]) * fd[ix, jx+1] - (v[jx-1] - u[ix]) * fd[ix, jx-1] ) * a[ix]
                    coll_diff = ( fd[ix, jx+1] - 2. * fd[ix, jx] + fd[ix, jx-1] )
                    
         
                    y[iy, jy] = fd[ix, jx] * self.grid.ht_inv \
                             + 0.5 * result_J1 * self.grid.hx_inv * self.grid.hv_inv \
                             - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                             - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        cdef npy.float64_t coll_drag, coll_diff
        
        self.get_data_arrays()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fp = self.da1.getLocalArray(F, self.localFp)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fh    = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v     = self.v
        cdef npy.ndarray[npy.float64_t, ndim=1] up    = self.up
        cdef npy.ndarray[npy.float64_t, ndim=1] ap    = self.ap
        cdef npy.ndarray[npy.float64_t, ndim=1] uh    = self.uh
        cdef npy.ndarray[npy.float64_t, ndim=1] ah    = self.ah
        
        
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = fp[ix, jx]
                    
                else:
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
                    
                    # collision operator
                    coll_drag = ( (v[jx+1] - up[ix]) * fp[ix, jx+1] - (v[jx-1] - up[ix]) * fp[ix, jx-1] ) * ap[ix] \
                              + ( (v[jx+1] - uh[ix]) * fh[ix, jx+1] - (v[jx-1] - uh[ix]) * fh[ix, jx-1] ) * ah[ix]
                    coll_diff = ( fp[ix, jx+1] - 2. * fp[ix, jx] + fp[ix, jx-1] ) \
                              + ( fh[ix, jx+1] - 2. * fh[ix, jx] + fh[ix, jx-1] )
                    
                    
                    y[iy, jy] = (fp[ix, jx] - fh[ix, jx]) * self.grid.ht_inv \
                             + result_J1 * self.grid.hx_inv * self.grid.hv_inv \
                             - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                             - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv
