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
        cdef npy.int64_t xe, xs
        
        (xs, xe), = self.da1.getRanges()
        
        self.get_data_arrays()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        
#         cdef npy.float64_t time_fac    = 0.
#         cdef npy.float64_t arak_fac_J1 = 0.
#         cdef npy.float64_t arak_fac_J2 = 0.
#         cdef npy.float64_t coll1_fac   = 0.
#         cdef npy.float64_t coll2_fac   = 0.
        
        cdef npy.float64_t time_fac    = 1.0  / self.ht
        cdef npy.float64_t arak_fac_J2 = 0.5 / (24. * self.hx * self.hv)
        
        cdef npy.float64_t coll1_fac   = - 0.5 * self.nu * 0.5 / self.hv
        cdef npy.float64_t coll2_fac   = - 0.5 * self.nu * self.hv2_inv
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        # Vlasov Equation
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            
            row.index = (i,)
                
            for j in range(0, self.nv):
                row.field = j
                
                # Dirichlet boundary conditions
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
                    A.setValueStencil(row, row, 1.0)
                    
                else:
                    for index, field, value in [
                            ((i-2,), j  , - (h_ave[ix-1, j+1] - h_ave[ix-1, j-1]) * arak_fac_J2),
                            ((i-1,), j-1, - (h_ave[ix-2, j  ] - h_ave[ix,   j-2]) * arak_fac_J2 \
                                          - (h_ave[ix-1, j+1] - h_ave[ix+1, j-1]) * arak_fac_J2),
                            ((i-1,), j+1, - (h_ave[ix,   j+2] - h_ave[ix-2, j  ]) * arak_fac_J2 \
                                          - (h_ave[ix+1, j+1] - h_ave[ix-1, j-1]) * arak_fac_J2),
                            ((i,  ), j-2, + (h_ave[ix+1, j-1] - h_ave[ix-1, j-1]) * arak_fac_J2),
                            ((i,  ), j-1, - coll1_fac * ( self.np[ix  ] * self.v[j-1] - self.up[ix  ] ) * self.ap[ix  ] \
                                          + coll2_fac),
                            ((i,  ), j  , time_fac \
                                          - 2. * coll2_fac),
                            ((i,  ), j+1, + coll1_fac * ( self.np[ix  ] * self.v[j+1] - self.up[ix  ] ) * self.ap[ix  ] \
                                          + coll2_fac),
                            ((i,  ), j+2, - (h_ave[ix+1, j+1] - h_ave[ix-1, j+1]) * arak_fac_J2),
                            ((i+1,), j-1, + (h_ave[ix+2, j  ] - h_ave[ix,   j-2]) * arak_fac_J2 \
                                          + (h_ave[ix+1, j+1] - h_ave[ix-1, j-1]) * arak_fac_J2),
                            ((i+1,), j+1, + (h_ave[ix,   j+2] - h_ave[ix+2, j  ]) * arak_fac_J2 \
                                          + (h_ave[ix-1, j+1] - h_ave[ix+1, j-1]) * arak_fac_J2),
                            ((i+2,), j,   + (h_ave[ix+1, j+1] - h_ave[ix+1, j-1]) * arak_fac_J2),
                        ]:

                        col.index = index
                        col.field = field
                        A.setValueStencil(row, col, value)
                        
        
        A.assemble()



    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy
        cdef npy.int64_t xe, xs
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        
        self.get_data_arrays()
        
        (xs, xe), = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fd = self.da1.getLocalArray(F, self.localFd)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(0, self.nv):
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fd[ix,j]
                    
                else:
                    ### TODO ###
                    ### collision operator not complete ###
                    ### TODO ###
#                     y[iy, j] = self.toolbox.time_derivative(f, ix, j) \
#                              + 0.5 * self.toolbox.arakawa_J2(f, h_ave, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fd, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fd, self.np, self.up, self.ep, self.ap, ix, j)
            
                    
                    jcc_J2 = (fd[ix+1, j+1] - fd[ix-1, j-1]) * (h_ave[ix-1, j+1] - h_ave[ix+1, j-1]) \
                           - (fd[ix-1, j+1] - fd[ix+1, j-1]) * (h_ave[ix+1, j+1] - h_ave[ix-1, j-1])
                    
                    jpc_J2 = fd[ix+2, j  ] * (h_ave[ix+1, j+1] - h_ave[ix+1, j-1]) \
                           - fd[ix-2, j  ] * (h_ave[ix-1, j+1] - h_ave[ix-1, j-1]) \
                           - fd[ix,   j+2] * (h_ave[ix+1, j+1] - h_ave[ix-1, j+1]) \
                           + fd[ix,   j-2] * (h_ave[ix+1, j-1] - h_ave[ix-1, j-1])
                    
                    jcp_J2 = fd[ix+1, j+1] * (h_ave[ix,   j+2] - h_ave[ix+2, j  ]) \
                           - fd[ix-1, j-1] * (h_ave[ix-2, j  ] - h_ave[ix,   j-2]) \
                           - fd[ix-1, j+1] * (h_ave[ix,   j+2] - h_ave[ix-2, j  ]) \
                           + fd[ix+1, j-1] * (h_ave[ix+2, j  ] - h_ave[ix,   j-2])
                    
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    
                    
                    y[iy, j] = fd[ix, j] * self.ht_inv \
                             + 0.5 * result_J2 * self.hx_inv * self.hv_inv \
                             + self.ht * self.regularisation * self.hx2_inv * ( 2. * fd[ix, j] - fd[ix+1, j] - fd[ix-1, j] ) \
                             + self.ht * self.regularisation * self.hv2_inv * ( 2. * fd[ix, j] - fd[ix, j+1] - fd[ix, j-1] )
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy
        cdef npy.int64_t xe, xs
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        
        self.get_data_arrays()
        
        (xs, xe), = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fp = self.da1.getLocalArray(F, self.localFp)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fh    = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(0, self.nv):
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fp[ix,j]
                    
                else:
#                     y[iy, j] = self.toolbox.time_derivative(fp, ix, j) \
#                              - self.toolbox.time_derivative(fh, ix, j) \
#                              + self.toolbox.arakawa_J2(f_ave, h_ave, ix, j) #\
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fp, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fh, self.nh, self.uh, self.eh, self.ah, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fp, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fh, self.nh, self.uh, self.eh, self.ah, ix, j)


                    jcc_J2 = (f_ave[ix+1, j+1] - f_ave[ix-1, j-1]) * (h_ave[ix-1, j+1] - h_ave[ix+1, j-1]) \
                           - (f_ave[ix-1, j+1] - f_ave[ix+1, j-1]) * (h_ave[ix+1, j+1] - h_ave[ix-1, j-1])
                    
                    jpc_J2 = f_ave[ix+2, j  ] * (h_ave[ix+1, j+1] - h_ave[ix+1, j-1]) \
                           - f_ave[ix-2, j  ] * (h_ave[ix-1, j+1] - h_ave[ix-1, j-1]) \
                           - f_ave[ix,   j+2] * (h_ave[ix+1, j+1] - h_ave[ix-1, j+1]) \
                           + f_ave[ix,   j-2] * (h_ave[ix+1, j-1] - h_ave[ix-1, j-1])
                    
                    jcp_J2 = f_ave[ix+1, j+1] * (h_ave[ix,   j+2] - h_ave[ix+2, j  ]) \
                           - f_ave[ix-1, j-1] * (h_ave[ix-2, j  ] - h_ave[ix,   j-2]) \
                           - f_ave[ix-1, j+1] * (h_ave[ix,   j+2] - h_ave[ix-2, j  ]) \
                           + f_ave[ix+1, j-1] * (h_ave[ix+2, j  ] - h_ave[ix,   j-2])
                    
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    
                    
                    y[iy, j] = (fp[ix, j] - fh[ix, j]) * self.ht_inv \
                             + result_J2 * self.hx_inv * self.hv_inv \
                             + self.ht * self.regularisation * self.hx2_inv * ( 2. * fp[ix, j] - fp[ix+1, j] - fp[ix-1, j] ) \
                             + self.ht * self.regularisation * self.hv2_inv * ( 2. * fp[ix, j] - fp[ix, j+1] - fp[ix, j-1] )
