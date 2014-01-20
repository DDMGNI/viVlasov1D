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
    Implements a variational integrator with first order
    finite-difference time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def formJacobian(self, Mat A):
        cdef npy.int64_t i, j, ix
        cdef npy.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.get_data_arrays()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] hh = self.h0 + self.h1h + self.h2h
        
        
#         cdef npy.float64_t time_fac    = 0.
#         cdef npy.float64_t arak_fac_J1 = 0.
#         cdef npy.float64_t arak_fac_J2 = 0.
#         cdef npy.float64_t coll1_fac   = 0.
#         cdef npy.float64_t coll2_fac   = 0.
        
        cdef npy.float64_t time_fac    = 1.0  / self.grid.ht
        cdef npy.float64_t arak_fac_J1 = + 1.0 / (12. * self.grid.hx * self.grid.hv)
        cdef npy.float64_t arak_fac_J2 = - 0.5 / (24. * self.grid.hx * self.grid.hv)
        
        cdef npy.float64_t coll1_fac   = - 0.5 * self.nu * 0.5 / self.grid.hv
        cdef npy.float64_t coll2_fac   = - 0.5 * self.nu * self.grid.hv2_inv
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        # Vlasov Equation
        for i in range(xs, xe):
            ix = i-xs+2
            
            row.index = (i,)
                
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                row.field = j
                
                # Dirichlet boundary conditions
                if j <= 1 or j >= self.grid.nv-2:
                    A.setValueStencil(row, row, 1.0)
                    
                else:
                    
                    for index, field, value in [
                            ((i-2,), j  , - (hh[ix-1, jx+1] - hh[ix-1, jx-1]) * arak_fac_J2),
                            ((i-1,), j-1, - (hh[ix-1, jx  ] - hh[ix,   jx-1]) * arak_fac_J1 \
                                          - (hh[ix-2, jx  ] - hh[ix,   jx-2]) * arak_fac_J2 \
                                          - (hh[ix-1, jx+1] - hh[ix+1, jx-1]) * arak_fac_J2),
                            ((i-1,), j  , - (hh[ix,   jx+1] - hh[ix,   jx-1]) * arak_fac_J1 \
                                          - (hh[ix-1, jx+1] - hh[ix-1, jx-1]) * arak_fac_J1),
                            ((i-1,), j+1, - (hh[ix,   jx+1] - hh[ix-1, jx  ]) * arak_fac_J1 \
                                          - (hh[ix,   jx+2] - hh[ix-2, jx  ]) * arak_fac_J2 \
                                          - (hh[ix+1, jx+1] - hh[ix-1, jx-1]) * arak_fac_J2),
                            ((i,  ), j-2, + (hh[ix+1, jx-1] - hh[ix-1, jx-1]) * arak_fac_J2),
                            ((i,  ), j-1, + (hh[ix+1, jx  ] - hh[ix-1, jx  ]) * arak_fac_J1 \
                                          + (hh[ix+1, jx-1] - hh[ix-1, jx-1]) * arak_fac_J1 \
                                          - coll1_fac * ( self.v[jx-1] - self.up[ix  ] ) * self.ap[ix  ] \
                                          + coll2_fac),
                            ((i,  ), j  , + time_fac \
                                          - 2. * coll2_fac),
                            ((i,  ), j+1, - (hh[ix+1, jx  ] - hh[ix-1, jx  ]) * arak_fac_J1 \
                                          - (hh[ix+1, jx+1] - hh[ix-1, jx+1]) * arak_fac_J1 \
                                          + coll1_fac * ( self.v[jx+1] - self.up[ix  ] ) * self.ap[ix  ] \
                                          + coll2_fac),
                            ((i,  ), j+2, - (hh[ix+1, jx+1] - hh[ix-1, jx+1]) * arak_fac_J2),
                            ((i+1,), j-1, + (hh[ix+1, jx  ] - hh[ix,   jx-1]) * arak_fac_J1 \
                                          + (hh[ix+2, jx  ] - hh[ix,   jx-2]) * arak_fac_J2 \
                                          + (hh[ix+1, jx+1] - hh[ix-1, jx-1]) * arak_fac_J2),
                            ((i+1,), j  , + (hh[ix,   jx+1] - hh[ix,   jx-1]) * arak_fac_J1 \
                                          + (hh[ix+1, jx+1] - hh[ix+1, jx-1]) * arak_fac_J1),
                            ((i+1,), j+1, + (hh[ix,   jx+1] - hh[ix+1, jx  ]) * arak_fac_J1 \
                                          + (hh[ix,   jx+2] - hh[ix+2, jx  ]) * arak_fac_J2 \
                                          + (hh[ix-1, jx+1] - hh[ix+1, jx-1]) * arak_fac_J2),
                            ((i+2,), j  , + (hh[ix+1, jx+1] - hh[ix+1, jx-1]) * arak_fac_J2),
                        ]:

                        col.index = index
                        col.field = field
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
        
        self.get_data_arrays()
#         self.get_data_arrays_jacobian()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fd = self.da1.getLocalArray(F, self.localFd)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] hh = self.h0 + self.h1h + self.h2h
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j <= 1 or j >= self.grid.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = fd[ix, jx]
                    
                else:
                    ### TODO ###
                    ### collision operator not complete ###
                    ### TODO ###
                    
                    jpp_J1 = (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (hh[ix,   jx+1] - hh[ix,   jx-1]) \
                           - (fd[ix,   jx+1] - fd[ix,   jx-1]) * (hh[ix+1, jx  ] - hh[ix-1, jx  ])
                    
                    jpc_J1 = fd[ix+1, jx  ] * (hh[ix+1, jx+1] - hh[ix+1, jx-1]) \
                           - fd[ix-1, jx  ] * (hh[ix-1, jx+1] - hh[ix-1, jx-1]) \
                           - fd[ix,   jx+1] * (hh[ix+1, jx+1] - hh[ix-1, jx+1]) \
                           + fd[ix,   jx-1] * (hh[ix+1, jx-1] - hh[ix-1, jx-1])
                    
                    jcp_J1 = fd[ix+1, jx+1] * (hh[ix,   jx+1] - hh[ix+1, jx  ]) \
                           - fd[ix-1, jx-1] * (hh[ix-1, jx  ] - hh[ix,   jx-1]) \
                           - fd[ix-1, jx+1] * (hh[ix,   jx+1] - hh[ix-1, jx  ]) \
                           + fd[ix+1, jx-1] * (hh[ix+1, jx  ] - hh[ix,   jx-1])
                    
                    jcc_J2 = (fd[ix+1, jx+1] - fd[ix-1, jx-1]) * (hh[ix-1, jx+1] - hh[ix+1, jx-1]) \
                           - (fd[ix-1, jx+1] - fd[ix+1, jx-1]) * (hh[ix+1, jx+1] - hh[ix-1, jx-1])
                    
                    jpc_J2 = fd[ix+2, jx  ] * (hh[ix+1, jx+1] - hh[ix+1, jx-1]) \
                           - fd[ix-2, jx  ] * (hh[ix-1, jx+1] - hh[ix-1, jx-1]) \
                           - fd[ix,   jx+2] * (hh[ix+1, jx+1] - hh[ix-1, jx+1]) \
                           + fd[ix,   jx-2] * (hh[ix+1, jx-1] - hh[ix-1, jx-1])
                    
                    jcp_J2 = fd[ix+1, jx+1] * (hh[ix,   jx+2] - hh[ix+2, jx  ]) \
                           - fd[ix-1, jx-1] * (hh[ix-2, jx  ] - hh[ix,   jx-2]) \
                           - fd[ix-1, jx+1] * (hh[ix,   jx+2] - hh[ix-2, jx  ]) \
                           + fd[ix+1, jx-1] * (hh[ix+2, jx  ] - hh[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
         
         
                    y[iy, jy] = fd[ix, jx] * self.grid.ht_inv \
                             + 0.5 * result_J4 * self.grid.hx_inv * self.grid.hv_inv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        
        self.get_data_arrays()
#         self.get_data_arrays_function()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fp = self.da1.getLocalArray(F, self.localFp)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fh = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] hp = self.h0 + self.h1p + self.h2p
        cdef npy.ndarray[npy.float64_t, ndim=2] hh = self.h0 + self.h1h + self.h2h
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j <= 1 or j >= self.grid.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = fp[ix, jx]
                    
                else:
                    y[iy, jy] = (fp[ix, jx] - fh[ix, jx]) * self.grid.ht_inv
                    

                    jpp_J1 = (fp[ix+1, jx  ] - fp[ix-1, jx  ]) * (hh[ix,   jx+1] - hh[ix,   jx-1]) \
                           - (fp[ix,   jx+1] - fp[ix,   jx-1]) * (hh[ix+1, jx  ] - hh[ix-1, jx  ])
                    
                    jpc_J1 = fp[ix+1, jx  ] * (hh[ix+1, jx+1] - hh[ix+1, jx-1]) \
                           - fp[ix-1, jx  ] * (hh[ix-1, jx+1] - hh[ix-1, jx-1]) \
                           - fp[ix,   jx+1] * (hh[ix+1, jx+1] - hh[ix-1, jx+1]) \
                           + fp[ix,   jx-1] * (hh[ix+1, jx-1] - hh[ix-1, jx-1])
                    
                    jcp_J1 = fp[ix+1, jx+1] * (hh[ix,   jx+1] - hh[ix+1, jx  ]) \
                           - fp[ix-1, jx-1] * (hh[ix-1, jx  ] - hh[ix,   jx-1]) \
                           - fp[ix-1, jx+1] * (hh[ix,   jx+1] - hh[ix-1, jx  ]) \
                           + fp[ix+1, jx-1] * (hh[ix+1, jx  ] - hh[ix,   jx-1])
                    
                    jcc_J2 = (fp[ix+1, jx+1] - fp[ix-1, jx-1]) * (hh[ix-1, jx+1] - hh[ix+1, jx-1]) \
                           - (fp[ix-1, jx+1] - fp[ix+1, jx-1]) * (hh[ix+1, jx+1] - hh[ix-1, jx-1])
                    
                    jpc_J2 = fp[ix+2, jx  ] * (hh[ix+1, jx+1] - hh[ix+1, jx-1]) \
                           - fp[ix-2, jx  ] * (hh[ix-1, jx+1] - hh[ix-1, jx-1]) \
                           - fp[ix,   jx+2] * (hh[ix+1, jx+1] - hh[ix-1, jx+1]) \
                           + fp[ix,   jx-2] * (hh[ix+1, jx-1] - hh[ix-1, jx-1])
                    
                    jcp_J2 = fp[ix+1, jx+1] * (hh[ix,   jx+2] - hh[ix+2, jx  ]) \
                           - fp[ix-1, jx-1] * (hh[ix-2, jx  ] - hh[ix,   jx-2]) \
                           - fp[ix-1, jx+1] * (hh[ix,   jx+2] - hh[ix-2, jx  ]) \
                           + fp[ix+1, jx-1] * (hh[ix+2, jx  ] - hh[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    y[iy, jy] += 0.5 * result_J4 * self.grid.hx_inv * self.grid.hv_inv
                    
                    
                    jpp_J1 = (fh[ix+1, jx  ] - fh[ix-1, jx  ]) * (hp[ix,   jx+1] - hp[ix,   jx-1]) \
                           - (fh[ix,   jx+1] - fh[ix,   jx-1]) * (hp[ix+1, jx  ] - hp[ix-1, jx  ])                    
                    
                    jpc_J1 = fh[ix+1, jx  ] * (hp[ix+1, jx+1] - hp[ix+1, jx-1]) \
                           - fh[ix-1, jx  ] * (hp[ix-1, jx+1] - hp[ix-1, jx-1]) \
                           - fh[ix,   jx+1] * (hp[ix+1, jx+1] - hp[ix-1, jx+1]) \
                           + fh[ix,   jx-1] * (hp[ix+1, jx-1] - hp[ix-1, jx-1])
                    
                    jcp_J1 = fh[ix+1, jx+1] * (hp[ix,   jx+1] - hp[ix+1, jx  ]) \
                           - fh[ix-1, jx-1] * (hp[ix-1, jx  ] - hp[ix,   jx-1]) \
                           - fh[ix-1, jx+1] * (hp[ix,   jx+1] - hp[ix-1, jx  ]) \
                           + fh[ix+1, jx-1] * (hp[ix+1, jx  ] - hp[ix,   jx-1])
                    
                    jcc_J2 = (fh[ix+1, jx+1] - fh[ix-1, jx-1]) * (hp[ix-1, jx+1] - hp[ix+1, jx-1]) \
                           - (fh[ix-1, jx+1] - fh[ix+1, jx-1]) * (hp[ix+1, jx+1] - hp[ix-1, jx-1])
                    
                    jpc_J2 = fh[ix+2, jx  ] * (hp[ix+1, jx+1] - hp[ix+1, jx-1]) \
                           - fh[ix-2, jx  ] * (hp[ix-1, jx+1] - hp[ix-1, jx-1]) \
                           - fh[ix,   jx+2] * (hp[ix+1, jx+1] - hp[ix-1, jx+1]) \
                           + fh[ix,   jx-2] * (hp[ix+1, jx-1] - hp[ix-1, jx-1])
                    
                    jcp_J2 = fh[ix+1, jx+1] * (hp[ix,   jx+2] - hp[ix+2, jx  ]) \
                           - fh[ix-1, jx-1] * (hp[ix-2, jx  ] - hp[ix,   jx-2]) \
                           - fh[ix-1, jx+1] * (hp[ix,   jx+2] - hp[ix-2, jx  ]) \
                           + fh[ix+1, jx-1] * (hp[ix+2, jx  ] - hp[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    y[iy, jy] += 0.5 * result_J4 * self.grid.hx_inv * self.grid.hv_inv
