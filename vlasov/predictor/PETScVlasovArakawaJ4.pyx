'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as npy
cimport numpy as npy

from petsc4py import PETSc

from vlasov.Toolbox import Toolbox


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
        cdef npy.int64_t xe, xs
        
        (xs, xe), = self.da1.getRanges()
        
        self.get_data_arrays()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] hh = self.h0 + self.h1h + self.h2h
        
        
#         cdef npy.float64_t time_fac    = 0.
#         cdef npy.float64_t arak_fac_J1 = 0.
#         cdef npy.float64_t arak_fac_J2 = 0.
#         cdef npy.float64_t coll1_fac   = 0.
#         cdef npy.float64_t coll2_fac   = 0.
        
        cdef npy.float64_t time_fac    = 1.0  / self.ht
        cdef npy.float64_t arak_fac_J1 = + 1.0 / (12. * self.hx * self.hv)
        cdef npy.float64_t arak_fac_J2 = - 0.5 / (24. * self.hx * self.hv)
        
        cdef npy.float64_t coll1_fac   = - 0.5 * self.nu * 0.5 / self.hv
        cdef npy.float64_t coll2_fac   = - 0.5 * self.nu * self.hv2_inv
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        # Vlasov Equation
        for i in range(xs, xe):
            ix = i-xs+2
            
            row.index = (i,)
                
            for j in range(0, self.nv):
                row.field = j
                
                # Dirichlet boundary conditions
                if j <= 1 or j >= self.nv-2:
                    A.setValueStencil(row, row, 1.0)
                    
                else:
                    
                    for index, field, value in [
                            ((i-2,), j  , - (hh[ix-1, j+1] - hh[ix-1, j-1]) * arak_fac_J2),
                            ((i-1,), j-1, - (hh[ix-1, j  ] - hh[ix,   j-1]) * arak_fac_J1 \
                                          - (hh[ix-2, j  ] - hh[ix,   j-2]) * arak_fac_J2 \
                                          - (hh[ix-1, j+1] - hh[ix+1, j-1]) * arak_fac_J2),
                            ((i-1,), j  , - (hh[ix,   j+1] - hh[ix,   j-1]) * arak_fac_J1 \
                                          - (hh[ix-1, j+1] - hh[ix-1, j-1]) * arak_fac_J1),
                            ((i-1,), j+1, - (hh[ix,   j+1] - hh[ix-1, j  ]) * arak_fac_J1 \
                                          - (hh[ix,   j+2] - hh[ix-2, j  ]) * arak_fac_J2 \
                                          - (hh[ix+1, j+1] - hh[ix-1, j-1]) * arak_fac_J2),
                            ((i,  ), j-2, + (hh[ix+1, j-1] - hh[ix-1, j-1]) * arak_fac_J2),
                            ((i,  ), j-1, + (hh[ix+1, j  ] - hh[ix-1, j  ]) * arak_fac_J1 \
                                          + (hh[ix+1, j-1] - hh[ix-1, j-1]) * arak_fac_J1 \
                                          - coll1_fac * ( self.v[j-1] - self.up[ix  ] ) * self.ap[ix  ] \
                                          + coll2_fac),
                            ((i,  ), j  , + time_fac \
                                          - 2. * coll2_fac),
                            ((i,  ), j+1, - (hh[ix+1, j  ] - hh[ix-1, j  ]) * arak_fac_J1 \
                                          - (hh[ix+1, j+1] - hh[ix-1, j+1]) * arak_fac_J1 \
                                          + coll1_fac * ( self.v[j+1] - self.up[ix  ] ) * self.ap[ix  ] \
                                          + coll2_fac),
                            ((i,  ), j+2, - (hh[ix+1, j+1] - hh[ix-1, j+1]) * arak_fac_J2),
                            ((i+1,), j-1, + (hh[ix+1, j  ] - hh[ix,   j-1]) * arak_fac_J1 \
                                          + (hh[ix+2, j  ] - hh[ix,   j-2]) * arak_fac_J2 \
                                          + (hh[ix+1, j+1] - hh[ix-1, j-1]) * arak_fac_J2),
                            ((i+1,), j  , + (hh[ix,   j+1] - hh[ix,   j-1]) * arak_fac_J1 \
                                          + (hh[ix+1, j+1] - hh[ix+1, j-1]) * arak_fac_J1),
                            ((i+1,), j+1, + (hh[ix,   j+1] - hh[ix+1, j  ]) * arak_fac_J1 \
                                          + (hh[ix,   j+2] - hh[ix+2, j  ]) * arak_fac_J2 \
                                          + (hh[ix-1, j+1] - hh[ix+1, j-1]) * arak_fac_J2),
                            ((i+2,), j  , + (hh[ix+1, j+1] - hh[ix+1, j-1]) * arak_fac_J2),
                        ]:

                        col.index = index
                        col.field = field
                        A.setValueStencil(row, col, value)
                        
        
        A.assemble()



    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(self, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy
        cdef npy.int64_t xe, xs
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        
        self.get_data_arrays()
#         self.get_data_arrays_jacobian()
        
        (xs, xe), = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] f  = self.fd
        cdef npy.ndarray[npy.float64_t, ndim=2] hh = self.h0 + self.h1h + self.h2h
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Vlasov equation
            for j in range(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = f[ix,j]
                    
                else:
                    ### TODO ###
                    ### collision operator not complete ###
                    ### TODO ###
                    
                    jpp_J1 = (f[ix+1, j  ] - f[ix-1, j  ]) * (hh[ix,   j+1] - hh[ix,   j-1]) \
                           - (f[ix,   j+1] - f[ix,   j-1]) * (hh[ix+1, j  ] - hh[ix-1, j  ])
                    
                    jpc_J1 = f[ix+1, j  ] * (hh[ix+1, j+1] - hh[ix+1, j-1]) \
                           - f[ix-1, j  ] * (hh[ix-1, j+1] - hh[ix-1, j-1]) \
                           - f[ix,   j+1] * (hh[ix+1, j+1] - hh[ix-1, j+1]) \
                           + f[ix,   j-1] * (hh[ix+1, j-1] - hh[ix-1, j-1])
                    
                    jcp_J1 = f[ix+1, j+1] * (hh[ix,   j+1] - hh[ix+1, j  ]) \
                           - f[ix-1, j-1] * (hh[ix-1, j  ] - hh[ix,   j-1]) \
                           - f[ix-1, j+1] * (hh[ix,   j+1] - hh[ix-1, j  ]) \
                           + f[ix+1, j-1] * (hh[ix+1, j  ] - hh[ix,   j-1])
                    
                    jcc_J2 = (f[ix+1, j+1] - f[ix-1, j-1]) * (hh[ix-1, j+1] - hh[ix+1, j-1]) \
                           - (f[ix-1, j+1] - f[ix+1, j-1]) * (hh[ix+1, j+1] - hh[ix-1, j-1])
                    
                    jpc_J2 = f[ix+2, j  ] * (hh[ix+1, j+1] - hh[ix+1, j-1]) \
                           - f[ix-2, j  ] * (hh[ix-1, j+1] - hh[ix-1, j-1]) \
                           - f[ix,   j+2] * (hh[ix+1, j+1] - hh[ix-1, j+1]) \
                           + f[ix,   j-2] * (hh[ix+1, j-1] - hh[ix-1, j-1])
                    
                    jcp_J2 = f[ix+1, j+1] * (hh[ix,   j+2] - hh[ix+2, j  ]) \
                           - f[ix-1, j-1] * (hh[ix-2, j  ] - hh[ix,   j-2]) \
                           - f[ix-1, j+1] * (hh[ix,   j+2] - hh[ix-2, j  ]) \
                           + f[ix+1, j-1] * (hh[ix+2, j  ] - hh[ix,   j-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
         
         
                    y[iy, j] = f[ix, j] * self.ht_inv \
                             + 0.5 * result_J4 * self.hx_inv * self.hv_inv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy
        cdef npy.int64_t xe, xs
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        
        self.get_data_arrays()
#         self.get_data_arrays_function()
        
        (xs, xe), = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fp = self.fd
        cdef npy.ndarray[npy.float64_t, ndim=2] fh = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] hp = self.h0 + self.h1p + self.h2p
        cdef npy.ndarray[npy.float64_t, ndim=2] hh = self.h0 + self.h1h + self.h2h
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Vlasov equation
            for j in range(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fp[ix,j]
                    
                else:
                    y[iy, j] = (fp[ix, j] - fh[ix, j]) * self.ht_inv
                    

                    jpp_J1 = (fp[ix+1, j  ] - fp[ix-1, j  ]) * (hh[ix,   j+1] - hh[ix,   j-1]) \
                           - (fp[ix,   j+1] - fp[ix,   j-1]) * (hh[ix+1, j  ] - hh[ix-1, j  ])
                    
                    jpc_J1 = fp[ix+1, j  ] * (hh[ix+1, j+1] - hh[ix+1, j-1]) \
                           - fp[ix-1, j  ] * (hh[ix-1, j+1] - hh[ix-1, j-1]) \
                           - fp[ix,   j+1] * (hh[ix+1, j+1] - hh[ix-1, j+1]) \
                           + fp[ix,   j-1] * (hh[ix+1, j-1] - hh[ix-1, j-1])
                    
                    jcp_J1 = fp[ix+1, j+1] * (hh[ix,   j+1] - hh[ix+1, j  ]) \
                           - fp[ix-1, j-1] * (hh[ix-1, j  ] - hh[ix,   j-1]) \
                           - fp[ix-1, j+1] * (hh[ix,   j+1] - hh[ix-1, j  ]) \
                           + fp[ix+1, j-1] * (hh[ix+1, j  ] - hh[ix,   j-1])
                    
                    jcc_J2 = (fp[ix+1, j+1] - fp[ix-1, j-1]) * (hh[ix-1, j+1] - hh[ix+1, j-1]) \
                           - (fp[ix-1, j+1] - fp[ix+1, j-1]) * (hh[ix+1, j+1] - hh[ix-1, j-1])
                    
                    jpc_J2 = fp[ix+2, j  ] * (hh[ix+1, j+1] - hh[ix+1, j-1]) \
                           - fp[ix-2, j  ] * (hh[ix-1, j+1] - hh[ix-1, j-1]) \
                           - fp[ix,   j+2] * (hh[ix+1, j+1] - hh[ix-1, j+1]) \
                           + fp[ix,   j-2] * (hh[ix+1, j-1] - hh[ix-1, j-1])
                    
                    jcp_J2 = fp[ix+1, j+1] * (hh[ix,   j+2] - hh[ix+2, j  ]) \
                           - fp[ix-1, j-1] * (hh[ix-2, j  ] - hh[ix,   j-2]) \
                           - fp[ix-1, j+1] * (hh[ix,   j+2] - hh[ix-2, j  ]) \
                           + fp[ix+1, j-1] * (hh[ix+2, j  ] - hh[ix,   j-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    y[iy, j] += 0.5 * result_J4 * self.hx_inv * self.hv_inv
                    
                    
                    jpp_J1 = (fh[ix+1, j  ] - fh[ix-1, j  ]) * (hp[ix,   j+1] - hp[ix,   j-1]) \
                           - (fh[ix,   j+1] - fh[ix,   j-1]) * (hp[ix+1, j  ] - hp[ix-1, j  ])                    
                    
                    jpc_J1 = fh[ix+1, j  ] * (hp[ix+1, j+1] - hp[ix+1, j-1]) \
                           - fh[ix-1, j  ] * (hp[ix-1, j+1] - hp[ix-1, j-1]) \
                           - fh[ix,   j+1] * (hp[ix+1, j+1] - hp[ix-1, j+1]) \
                           + fh[ix,   j-1] * (hp[ix+1, j-1] - hp[ix-1, j-1])
                    
                    jcp_J1 = fh[ix+1, j+1] * (hp[ix,   j+1] - hp[ix+1, j  ]) \
                           - fh[ix-1, j-1] * (hp[ix-1, j  ] - hp[ix,   j-1]) \
                           - fh[ix-1, j+1] * (hp[ix,   j+1] - hp[ix-1, j  ]) \
                           + fh[ix+1, j-1] * (hp[ix+1, j  ] - hp[ix,   j-1])
                    
                    jcc_J2 = (fh[ix+1, j+1] - fh[ix-1, j-1]) * (hp[ix-1, j+1] - hp[ix+1, j-1]) \
                           - (fh[ix-1, j+1] - fh[ix+1, j-1]) * (hp[ix+1, j+1] - hp[ix-1, j-1])
                    
                    jpc_J2 = fh[ix+2, j  ] * (hp[ix+1, j+1] - hp[ix+1, j-1]) \
                           - fh[ix-2, j  ] * (hp[ix-1, j+1] - hp[ix-1, j-1]) \
                           - fh[ix,   j+2] * (hp[ix+1, j+1] - hp[ix-1, j+1]) \
                           + fh[ix,   j-2] * (hp[ix+1, j-1] - hp[ix-1, j-1])
                    
                    jcp_J2 = fh[ix+1, j+1] * (hp[ix,   j+2] - hp[ix+2, j  ]) \
                           - fh[ix-1, j-1] * (hp[ix-2, j  ] - hp[ix,   j-2]) \
                           - fh[ix-1, j+1] * (hp[ix,   j+2] - hp[ix-2, j  ]) \
                           + fh[ix+1, j-1] * (hp[ix+2, j  ] - hp[ix,   j-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    y[iy, j] += 0.5 * result_J4 * self.hx_inv * self.hv_inv
