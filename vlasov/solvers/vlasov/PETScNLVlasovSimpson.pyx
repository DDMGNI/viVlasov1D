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
    def jacobian(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t bracket, coll_drag, coll_diff
        cdef npy.float64_t bracket11, bracket12, bracket21, bracket22
        
        self.get_data_arrays()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fd = self.da1.getLocalArray(F, self.localFd)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v = self.v
        cdef npy.ndarray[npy.float64_t, ndim=1] u = self.up
        cdef npy.ndarray[npy.float64_t, ndim=1] a = self.ap
        
        
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
                    
                    bracket22 = ( \
                                  + (fd[ix,   jx+2] - fd[ix,   jx-2]) * (h_ave[ix-2, jx  ] - h_ave[ix+2, jx  ]) \
                                  + (fd[ix+2, jx  ] - fd[ix-2, jx  ]) * (h_ave[ix,   jx+2] - h_ave[ix,   jx-2]) \
                                  + fd[ix,   jx+2] * (h_ave[ix-2, jx+2] - h_ave[ix+2, jx+2]) \
                                  + fd[ix,   jx-2] * (h_ave[ix+2, jx-2] - h_ave[ix-2, jx-2]) \
                                  + fd[ix+2, jx  ] * (h_ave[ix+2, jx+2] - h_ave[ix+2, jx-2]) \
                                  + fd[ix-2, jx  ] * (h_ave[ix-2, jx-2] - h_ave[ix-2, jx+2]) \
                                  + fd[ix+2, jx+2] * (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) \
                                  + fd[ix+2, jx-2] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2]) \
                                  + fd[ix-2, jx+2] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx+2]) \
                                  + fd[ix-2, jx-2] * (h_ave[ix,   jx-2] - h_ave[ix-2, jx  ]) \
                                ) / 48.
                    
                    bracket12 = ( \
                                  + (fd[ix,   jx+2] - fd[ix,   jx-2]) * (h_ave[ix-1, jx  ] - h_ave[ix+1, jx  ]) \
                                  + (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (h_ave[ix,   jx+2] - h_ave[ix,   jx-2]) \
                                  + fd[ix,   jx+2] * (h_ave[ix-1, jx+2] - h_ave[ix+1, jx+2]) \
                                  + fd[ix,   jx-2] * (h_ave[ix+1, jx-2] - h_ave[ix-1, jx-2]) \
                                  + fd[ix+1, jx  ] * (h_ave[ix+1, jx+2] - h_ave[ix+1, jx-2]) \
                                  + fd[ix-1, jx  ] * (h_ave[ix-1, jx-2] - h_ave[ix-1, jx+2]) \
                                  + fd[ix+1, jx+2] * (h_ave[ix,   jx+2] - h_ave[ix+1, jx  ]) \
                                  + fd[ix+1, jx-2] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-2]) \
                                  + fd[ix-1, jx-2] * (h_ave[ix,   jx-2] - h_ave[ix-1, jx  ]) \
                                  + fd[ix-1, jx+2] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx+2]) \
                                ) / 24.
                    
                    bracket21 = ( \
                                  + (fd[ix,   jx+1] - fd[ix,   jx-1]) * (h_ave[ix-2, jx  ] - h_ave[ix+2, jx  ]) \
                                  + (fd[ix+2, jx  ] - fd[ix-2, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                                  + fd[ix,   jx+1] * (h_ave[ix-2, jx+1] - h_ave[ix+2, jx+1]) \
                                  + fd[ix,   jx-1] * (h_ave[ix+2, jx-1] - h_ave[ix-2, jx-1]) \
                                  + fd[ix+2, jx  ] * (h_ave[ix+2, jx+1] - h_ave[ix+2, jx-1]) \
                                  + fd[ix-2, jx  ] * (h_ave[ix-2, jx-1] - h_ave[ix-2, jx+1]) \
                                  + fd[ix+2, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+2, jx  ]) \
                                  + fd[ix+2, jx-1] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-1]) \
                                  + fd[ix-2, jx+1] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx+1]) \
                                  + fd[ix-2, jx-1] * (h_ave[ix,   jx-1] - h_ave[ix-2, jx  ]) \
                                ) / 24.
                    
                    bracket11 = ( \
                                  + (fd[ix,   jx+1] - fd[ix,   jx-1]) * (h_ave[ix-1, jx  ] - h_ave[ix+1, jx  ]) \
                                  + (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                                  + fd[ix,   jx+1] * (h_ave[ix-1, jx+1] - h_ave[ix+1, jx+1]) \
                                  + fd[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) \
                                  + fd[ix-1, jx  ] * (h_ave[ix-1, jx-1] - h_ave[ix-1, jx+1]) \
                                  + fd[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                                  + fd[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                                  + fd[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1]) \
                                  + fd[ix-1, jx-1] * (h_ave[ix,   jx-1] - h_ave[ix-1, jx  ]) \
                                  + fd[ix-1, jx+1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx+1]) \
                                ) / 12.
                    
                    bracket = ( 25. * bracket11 - 10. * bracket12 - 10. * bracket21 + 4. * bracket22 ) / 9.
                    
                    
                    # collision operator
                    coll_drag = ( (v[jx+1] - u[ix]) * fd[ix, jx+1] - (v[jx-1] - u[ix]) * fd[ix, jx-1] ) * a[ix]
                    coll_diff = ( fd[ix, jx+1] - 2. * fd[ix, jx] + fd[ix, jx-1] )
                    
         
                    y[iy, jy] = fd[ix, jx] * self.grid.ht_inv \
                              + 0.5 * bracket * self.grid.hx_inv * self.grid.hv_inv \
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
        cdef npy.float64_t bracket, coll_drag, coll_diff
        cdef npy.float64_t bracket11, bracket12, bracket21, bracket22
        
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
                    bracket22 = ( \
                                  + (f_ave[ix,   jx+2] - f_ave[ix,   jx-2]) * (h_ave[ix-2, jx  ] - h_ave[ix+2, jx  ]) \
                                  + (f_ave[ix+2, jx  ] - f_ave[ix-2, jx  ]) * (h_ave[ix,   jx+2] - h_ave[ix,   jx-2]) \
                                  + f_ave[ix,   jx+2] * (h_ave[ix-2, jx+2] - h_ave[ix+2, jx+2]) \
                                  + f_ave[ix,   jx-2] * (h_ave[ix+2, jx-2] - h_ave[ix-2, jx-2]) \
                                  + f_ave[ix+2, jx  ] * (h_ave[ix+2, jx+2] - h_ave[ix+2, jx-2]) \
                                  + f_ave[ix-2, jx  ] * (h_ave[ix-2, jx-2] - h_ave[ix-2, jx+2]) \
                                  + f_ave[ix+2, jx+2] * (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) \
                                  + f_ave[ix+2, jx-2] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2]) \
                                  + f_ave[ix-2, jx+2] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx+2]) \
                                  + f_ave[ix-2, jx-2] * (h_ave[ix,   jx-2] - h_ave[ix-2, jx  ]) \
                                ) / 48.
                    
                    bracket12 = ( \
                                  + (f_ave[ix,   jx+2] - f_ave[ix,   jx-2]) * (h_ave[ix-1, jx  ] - h_ave[ix+1, jx  ]) \
                                  + (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ]) * (h_ave[ix,   jx+2] - h_ave[ix,   jx-2]) \
                                  + f_ave[ix,   jx+2] * (h_ave[ix-1, jx+2] - h_ave[ix+1, jx+2]) \
                                  + f_ave[ix,   jx-2] * (h_ave[ix+1, jx-2] - h_ave[ix-1, jx-2]) \
                                  + f_ave[ix+1, jx  ] * (h_ave[ix+1, jx+2] - h_ave[ix+1, jx-2]) \
                                  + f_ave[ix-1, jx  ] * (h_ave[ix-1, jx-2] - h_ave[ix-1, jx+2]) \
                                  + f_ave[ix+1, jx+2] * (h_ave[ix,   jx+2] - h_ave[ix+1, jx  ]) \
                                  + f_ave[ix+1, jx-2] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-2]) \
                                  + f_ave[ix-1, jx-2] * (h_ave[ix,   jx-2] - h_ave[ix-1, jx  ]) \
                                  + f_ave[ix-1, jx+2] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx+2]) \
                                ) / 24.
                    
                    bracket21 = ( \
                                  + (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) * (h_ave[ix-2, jx  ] - h_ave[ix+2, jx  ]) \
                                  + (f_ave[ix+2, jx  ] - f_ave[ix-2, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                                  + f_ave[ix,   jx+1] * (h_ave[ix-2, jx+1] - h_ave[ix+2, jx+1]) \
                                  + f_ave[ix,   jx-1] * (h_ave[ix+2, jx-1] - h_ave[ix-2, jx-1]) \
                                  + f_ave[ix+2, jx  ] * (h_ave[ix+2, jx+1] - h_ave[ix+2, jx-1]) \
                                  + f_ave[ix-2, jx  ] * (h_ave[ix-2, jx-1] - h_ave[ix-2, jx+1]) \
                                  + f_ave[ix+2, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+2, jx  ]) \
                                  + f_ave[ix+2, jx-1] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-1]) \
                                  + f_ave[ix-2, jx+1] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx+1]) \
                                  + f_ave[ix-2, jx-1] * (h_ave[ix,   jx-1] - h_ave[ix-2, jx  ]) \
                                ) / 24.
                    
                    bracket11 = ( \
                                  + (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) * (h_ave[ix-1, jx  ] - h_ave[ix+1, jx  ]) \
                                  + (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                                  + f_ave[ix,   jx+1] * (h_ave[ix-1, jx+1] - h_ave[ix+1, jx+1]) \
                                  + f_ave[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) \
                                  + f_ave[ix-1, jx  ] * (h_ave[ix-1, jx-1] - h_ave[ix-1, jx+1]) \
                                  + f_ave[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                                  + f_ave[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                                  + f_ave[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1]) \
                                  + f_ave[ix-1, jx-1] * (h_ave[ix,   jx-1] - h_ave[ix-1, jx  ]) \
                                  + f_ave[ix-1, jx+1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx+1]) \
                                ) / 12.
                    
                    bracket = ( 25. * bracket11 - 10. * bracket12 - 10. * bracket21 + 4. * bracket22 ) / 9.
                    
                    
                    # collision operator
                    coll_drag = ( (v[jx+1] - up[ix]) * fp[ix, jx+1] - (v[jx-1] - up[ix]) * fp[ix, jx-1] ) * ap[ix] \
                              + ( (v[jx+1] - uh[ix]) * fh[ix, jx+1] - (v[jx-1] - uh[ix]) * fh[ix, jx-1] ) * ah[ix]
                    coll_diff = ( fp[ix, jx+1] - 2. * fp[ix, jx] + fp[ix, jx-1] ) \
                              + ( fh[ix, jx+1] - 2. * fh[ix, jx] + fh[ix, jx-1] )
                    
                    
                    y[iy, jy] = (fp[ix, jx] - fh[ix, jx]) * self.grid.ht_inv \
                              + 1.0 * bracket * self.grid.hx_inv * self.grid.hv_inv \
                              - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                              - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv
