'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as npy
cimport numpy as npy

from petsc4py import PETSc

from vlasov.toolbox.Toolbox import Toolbox


cdef class PETScVlasovSolverPotential(PETScVlasovSolverBase):
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
        
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (self.fd + self.fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = 0.5 * (self.h1p + self.h1h)
        
        
#         cdef npy.float64_t time_fac    = 0.
#         cdef npy.float64_t arak_fac_J1 = 0.
#         cdef npy.float64_t arak_fac_J2 = 0.
        
        cdef npy.float64_t time_fac    = 1.0  / self.grid.ht
        cdef npy.float64_t arak_fac_J1 = + 1.0 / (12. * self.grid.hx * self.grid.hv)
        cdef npy.float64_t arak_fac_J2 = - 0.5 / (24. * self.grid.hx * self.grid.hv)
        
        
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
                    
                    for index, field, value in [
                            ((i-2,), j  , - (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
                            ((i-1,), j-1, - (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) * arak_fac_J1 \
                                          - (h_ave[ix-2, jx  ] - h_ave[ix,   jx-2]) * arak_fac_J2 \
                                          - (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J2),
                            ((i-1,), j  , - (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) * arak_fac_J1 \
                                          - (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J1),
                            ((i-1,), j+1, - (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
                                          - (h_ave[ix,   jx+2] - h_ave[ix-2, jx  ]) * arak_fac_J2 \
                                          - (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
                            ((i,  ), j-2, + (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
                            ((i,  ), j-1, + (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
                                          + (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) * arak_fac_J1),
                            ((i,  ), j  , + time_fac),
                            ((i,  ), j+1, - (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
                                          - (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) * arak_fac_J1),
                            ((i,  ), j+2, - (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) * arak_fac_J2),
                            ((i+1,), j-1, + (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1]) * arak_fac_J1 \
                                          + (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2]) * arak_fac_J2 \
                                          + (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
                            ((i+1,), j  , + (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) * arak_fac_J1 \
                                          + (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J1),
                            ((i+1,), j+1, + (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) * arak_fac_J1 \
                                          + (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) * arak_fac_J2 \
                                          + (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J2),
                            ((i+2,), j  , + (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J2),
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
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] f = self.da1.getLocalArray(F, self.localFd)
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = 0.5 * (self.h1p + self.h1h)
        
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = f[ix, jx]
                    
                else:
                    
                    jpp_J1 = (f[ix+1, jx  ] - f[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                           - (f[ix,   jx+1] - f[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                    
                    jpc_J1 = f[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - f[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - f[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + f[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J1 = f[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                           - f[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                           - f[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                           + f[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                    
                    jcc_J2 = (f[ix+1, jx+1] - f[ix-1, jx-1]) * (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) \
                           - (f[ix-1, jx+1] - f[ix+1, jx-1]) * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1])
                    
                    jpc_J2 = f[ix+2, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - f[ix-2, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - f[ix,   jx+2] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + f[ix,   jx-2] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J2 = f[ix+1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) \
                           - f[ix-1, jx-1] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx-2]) \
                           - f[ix-1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix-2, jx  ]) \
                           + f[ix+1, jx-1] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = result_J1 - 0.5 * result_J2
         
         
                    y[iy, jy] = f[ix, jx] * self.grid.ht_inv \
                             + result_J4 * self.grid.hx_inv * self.grid.hv_inv
    
    
    
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
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fp = self.da1.getLocalArray(F, self.localFp)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fh    = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = 0.5 * (self.h1p + self.h1h)
        
        
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
                    
                    jcc_J2 = (f_ave[ix+1, jx+1] - f_ave[ix-1, jx-1]) * (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) \
                           - (f_ave[ix-1, jx+1] - f_ave[ix+1, jx-1]) * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1])
                    
                    jpc_J2 = f_ave[ix+2, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - f_ave[ix-2, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - f_ave[ix,   jx+2] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + f_ave[ix,   jx-2] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J2 = f_ave[ix+1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) \
                           - f_ave[ix-1, jx-1] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx-2]) \
                           - f_ave[ix-1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix-2, jx  ]) \
                           + f_ave[ix+1, jx-1] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    
                    y[iy, jy] = (fp[ix, jx] - fh[ix, jx]) * self.grid.ht_inv \
                             + result_J4 * self.grid.hx_inv * self.grid.hv_inv
