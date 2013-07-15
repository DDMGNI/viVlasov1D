'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as npy
cimport numpy as npy

from petsc4py import PETSc

from vlasov.toolbox.Toolbox import Toolbox


cdef class PETScSolver(PETScFullSolverBase):
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
        
        (xs, xe), = self.da2.getRanges()
        
        self.get_data_arrays()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (self.fp + self.fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        
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
        
        
        # Poisson equation
        for i in range(xs, xe):
            row.index = (i,)
            row.field = self.nv
            
            # charge density
            col.index = (i,  )
            col.field = self.nv+1
            A.setValueStencil(row, col, self.charge)
            
            
            # Laplace operator
            for index, value in [
                    ((i-1,), - 1. * self.hx2_inv),
                    ((i,  ), + 2. * self.hx2_inv),
                    ((i+1,), - 1. * self.hx2_inv),
                ]:
                
                col.index = index
                col.field = self.nv
                A.setValueStencil(row, col, value)
         
            
        
        # moments
        for i in range(xs, xe):
            ix = i-xs+2
            
            row.index = (i,)
            col.index = (i,)
            
            
            # density
            row.field = self.nv+1
            col.field = self.nv+1
            
            A.setValueStencil(row, col, 1.)
            
            for j in range(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - 1. * self.hv)
             
            
            # average velocity density
            row.field = self.nv+2
            col.field = self.nv+2
            
            A.setValueStencil(row, col, 1.)
            
            for j in range(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - self.v[j] * self.hv)
            
            
            # average energy density
            row.field = self.nv+3
            col.field = self.nv+3
            
            A.setValueStencil(row, col, 1.)
            
            for j in range(0, self.nv):
                col.field = j
                A.setValueStencil(row, col, - self.v[j]**2 * self.hv)
                
            
        
        # Vlasov Equation
        for i in range(xs, xe):
            ix = i-xs+2
            
            row.index = (i,)
#             col.index = (i,)
                
            for j in range(0, self.nv):
                row.field = j
                
                # Dirichlet boundary conditions
                if j <= 1 or j >= self.nv-2:
                    A.setValueStencil(row, row, 1.0)
                    
                else:
                    
                    for index, field, value in [
                            ((i-2,), j  , - (h_ave[ix-1, j+1] - h_ave[ix-1, j-1]) * arak_fac_J2),
                            ((i-1,), j-1, - (h_ave[ix-1, j  ] - h_ave[ix,   j-1]) * arak_fac_J1 \
                                          - (h_ave[ix-2, j  ] - h_ave[ix,   j-2]) * arak_fac_J2 \
                                          - (h_ave[ix-1, j+1] - h_ave[ix+1, j-1]) * arak_fac_J2),
                            ((i-1,), j  , - (h_ave[ix,   j+1] - h_ave[ix,   j-1]) * arak_fac_J1 \
                                          - (h_ave[ix-1, j+1] - h_ave[ix-1, j-1]) * arak_fac_J1),
                            ((i-1,), j+1, - (h_ave[ix,   j+1] - h_ave[ix-1, j  ]) * arak_fac_J1 \
                                          - (h_ave[ix,   j+2] - h_ave[ix-2, j  ]) * arak_fac_J2 \
                                          - (h_ave[ix+1, j+1] - h_ave[ix-1, j-1]) * arak_fac_J2),
                            ((i,  ), j-2, + (h_ave[ix+1, j-1] - h_ave[ix-1, j-1]) * arak_fac_J2),
                            ((i,  ), j-1, + (h_ave[ix+1, j  ] - h_ave[ix-1, j  ]) * arak_fac_J1 \
                                          + (h_ave[ix+1, j-1] - h_ave[ix-1, j-1]) * arak_fac_J1 \
                                          - coll1_fac * ( self.v[j-1] - self.up[ix  ] ) * self.ap[ix  ] \
                                          + coll2_fac),
                            ((i,  ), j  , + time_fac \
                                          - 2. * coll2_fac),
                            ((i,  ), j+1, - (h_ave[ix+1, j  ] - h_ave[ix-1, j  ]) * arak_fac_J1 \
                                          - (h_ave[ix+1, j+1] - h_ave[ix-1, j+1]) * arak_fac_J1 \
                                          + coll1_fac * ( self.v[j+1] - self.up[ix  ] ) * self.ap[ix  ] \
                                          + coll2_fac),
                            ((i,  ), j+2, - (h_ave[ix+1, j+1] - h_ave[ix-1, j+1]) * arak_fac_J2),
                            ((i+1,), j-1, + (h_ave[ix+1, j  ] - h_ave[ix,   j-1]) * arak_fac_J1 \
                                          + (h_ave[ix+2, j  ] - h_ave[ix,   j-2]) * arak_fac_J2 \
                                          + (h_ave[ix+1, j+1] - h_ave[ix-1, j-1]) * arak_fac_J2),
                            ((i+1,), j  , + (h_ave[ix,   j+1] - h_ave[ix,   j-1]) * arak_fac_J1 \
                                          + (h_ave[ix+1, j+1] - h_ave[ix+1, j-1]) * arak_fac_J1),
                            ((i+1,), j+1, + (h_ave[ix,   j+1] - h_ave[ix+1, j  ]) * arak_fac_J1 \
                                          + (h_ave[ix,   j+2] - h_ave[ix+2, j  ]) * arak_fac_J2 \
                                          + (h_ave[ix-1, j+1] - h_ave[ix+1, j-1]) * arak_fac_J2),
                            ((i+2,), j  , + (h_ave[ix+1, j+1] - h_ave[ix+1, j-1]) * arak_fac_J2),
                            
                            ((i-2,), self.nv,    + 1. * (f_ave[ix-1, j+1] - f_ave[ix-1, j-1]) * arak_fac_J2),
                            ((i-1,), self.nv,    + 2. * (f_ave[ix,   j+1] - f_ave[ix,   j-1]) * arak_fac_J1 \
                                                 + 1. * (f_ave[ix-1, j+1] - f_ave[ix-1, j-1]) * arak_fac_J1 \
                                                 + 1. * (f_ave[ix-1, j+1] - f_ave[ix+1, j-1]) * arak_fac_J2 \
                                                 + 1. * (f_ave[ix+1, j+1] - f_ave[ix-1, j-1]) * arak_fac_J2 \
                                                 + 1. * (f_ave[ix-2, j  ] - f_ave[ix,   j-2]) * arak_fac_J2 \
                                                 + 1. * (f_ave[ix,   j+2] - f_ave[ix-2, j  ]) * arak_fac_J2),
                            ((i,  ), self.nv,    - 1. * (f_ave[ix+1, j-1] - f_ave[ix-1, j-1]) * arak_fac_J1 \
                                                 + 1. * (f_ave[ix+1, j+1] - f_ave[ix-1, j+1]) * arak_fac_J1 \
                                                 + 1. * (f_ave[ix+1, j+1] - f_ave[ix-1, j+1]) * arak_fac_J2 \
                                                 - 1. * (f_ave[ix+1, j-1] - f_ave[ix-1, j-1]) * arak_fac_J2),
                            ((i+1,), self.nv,    - 2. * (f_ave[ix,   j+1] - f_ave[ix,   j-1]) * arak_fac_J1 \
                                                 - 1. * (f_ave[ix+1, j+1] - f_ave[ix+1, j-1]) * arak_fac_J1 \
                                                 - 1. * (f_ave[ix-1, j+1] - f_ave[ix+1, j-1]) * arak_fac_J2 \
                                                 - 1. * (f_ave[ix+1, j+1] - f_ave[ix-1, j-1]) * arak_fac_J2 \
                                                 - 1. * (f_ave[ix,   j+2] - f_ave[ix+2, j  ]) * arak_fac_J2 \
                                                 - 1. * (f_ave[ix+2, j  ] - f_ave[ix,   j-2]) * arak_fac_J2),
                            ((i+2,), self.nv,    - 1. * (f_ave[ix+1, j+1] - f_ave[ix+1, j-1]) * arak_fac_J2),
                            
                            
#                             ### TODO ###
#                             check the following lines
#                             not updodate anymore
#                             ### TODO ###
#                             ((i,  ), self.nv+1,  + coll1_fac * self.fp[ix,   j+1] * self.v[j+1] * self.ap[ix  ] \
#                                                  - coll1_fac * self.fp[ix,   j-1] * self.v[j-1] * self.ap[ix  ] \
#                                                  + coll1_fac * self.fp[ix,   j+1] * ( self.np[ix  ] * self.v[j+1] - self.up[ix  ] ) * ( self.ap[ix  ] / self.np[ix  ] - self.ep[ix  ] * self.ap[ix  ]**2 / self.np[ix  ] ) \
#                                                  - coll1_fac * self.fp[ix,   j-1] * ( self.np[ix  ] * self.v[j-1] - self.up[ix  ] ) * ( self.ap[ix  ] / self.np[ix  ] - self.ep[ix  ] * self.ap[ix  ]**2 / self.np[ix  ] ) ),
#                              
#                             ((i,  ), self.nv+2,  - coll1_fac * self.fp[ix,   j+1] * self.ap[ix  ] \
#                                                  + coll1_fac * self.fp[ix,   j-1] * self.ap[ix  ] \
#                                                  + coll1_fac * self.fp[ix,   j+1] * ( self.np[ix  ] * self.v[j+1] - self.up[ix  ] ) * 2. * self.up[ix  ] * self.ap[ix  ]**2 / self.np[ix  ] \
#                                                  - coll1_fac * self.fp[ix,   j-1] * ( self.np[ix  ] * self.v[j-1] - self.up[ix  ] ) * 2. * self.up[ix  ] * self.ap[ix  ]**2 / self.np[ix  ] ),
#                              
#                             ((i,  ), self.nv+3,  - coll1_fac * self.fp[ix,   j+1] * ( self.np[ix  ] * self.v[j+1] - self.up[ix  ] ) * self.ap[ix  ]**2 \
#                                                  + coll1_fac * self.fp[ix,   j-1] * ( self.np[ix  ] * self.v[j-1] - self.up[ix  ] ) * self.ap[ix  ]**2 ),
                        ]:

                        col.index = index
                        col.field = field
                        A.setValueStencil(row, col, value)
                        
        
        A.assemble()



    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(self, Vec Y):
        cdef npy.uint64_t i, j
        cdef npy.uint64_t ix, iy
        cdef npy.uint64_t xe, xs
        
        self.toolbox.compute_density(self.Fd, self.Nc)
        self.toolbox.compute_velocity_density(self.Fd, self.Uc)
        self.toolbox.compute_energy_density(self.Fd, self.Ec)
        
        self.get_data_arrays()
        
        (xs, xe), = self.da2.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da2.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (self.fp + self.fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) + 0.5 * (self.h2p + self.h2h)
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Poisson equation
            y[iy, self.nv] = - ( self.pd[ix-1] + self.pd[ix+1] - 2. * self.pd[ix] ) * self.hx2_inv + self.charge * self.nd[ix]
            
            
            # moments
            y[iy, self.nv+1] = self.nd[ix] - self.nc[ix]
            y[iy, self.nv+2] = self.ud[ix] - self.uc[ix]
            y[iy, self.nv+3] = self.ed[ix] - self.ec[ix]
            
            
            # Vlasov equation
            for j in range(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = self.fd[ix,j]
                    
                else:
                    ### TODO ###
                    ### collision operator not complete ###
                    ### TODO ###
                    y[iy, j] = self.toolbox.time_derivative(self.fd, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(self.fd, h_ave,      ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(f_ave,   self.h1d,   ix, j) \
                             + self.ht * self.regularisation * self.hx2_inv * ( 2. * self.fd[ix, j] - self.fd[ix+1, j] - self.fd[ix-1, j] ) \
                             + self.ht * self.regularisation * self.hv2_inv * ( 2. * self.fd[ix, j] - self.fd[ix, j+1] - self.fd[ix, j-1] ) #\
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fd, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fd, self.np, self.up, self.ep, self.ap, ix, j)
            
            
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec Y):
        cdef npy.uint64_t i, j
        cdef npy.uint64_t ix, iy
        cdef npy.uint64_t xe, xs
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        
        cdef npy.float64_t nmean = self.Nd.sum() / self.nx
        
        self.toolbox.compute_density(self.Fd, self.Nc)
        self.toolbox.compute_velocity_density(self.Fd, self.Uc)
        self.toolbox.compute_energy_density(self.Fd, self.Ec)
        
        self.get_data_arrays()
        
        (xs, xe), = self.da2.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da2.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=1] pp    = self.pd
        cdef npy.ndarray[npy.float64_t, ndim=1] np    = self.nd
        cdef npy.ndarray[npy.float64_t, ndim=1] up    = self.ud
        cdef npy.ndarray[npy.float64_t, ndim=1] ep    = self.ed
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fp    = self.fd
        cdef npy.ndarray[npy.float64_t, ndim=2] fh    = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1d + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Poisson equation
            y[iy, self.nv] = - ( pp[ix-1] + pp[ix+1] - 2. * pp[ix] ) * self.hx2_inv + self.charge * (np[ix] - nmean)
            
            
            # moments
            y[iy, self.nv+1] = np[ix] - self.nc[ix]
            y[iy, self.nv+2] = up[ix] - self.uc[ix]
            y[iy, self.nv+3] = ep[ix] - self.ec[ix]
            
            
            # Vlasov equation
            for j in range(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fp[ix,j]
                    
                else:
#                     y[iy, j] = self.toolbox.time_derivative(self.fd, ix, j) \
#                              - self.toolbox.time_derivative(self.fh, ix, j) \
#                              + self.toolbox.arakawa_J4(f_ave, h_ave, ix, j) #\
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fp, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fh, self.nh, self.uh, self.eh, self.ah, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fp, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fh, self.nh, self.uh, self.eh, self.ah, ix, j)

                    jpp_J1 = (f_ave[ix+1, j  ] - f_ave[ix-1, j  ]) * (h_ave[ix,   j+1] - h_ave[ix,   j-1]) \
                           - (f_ave[ix,   j+1] - f_ave[ix,   j-1]) * (h_ave[ix+1, j  ] - h_ave[ix-1, j  ])
                    
                    jpc_J1 = f_ave[ix+1, j  ] * (h_ave[ix+1, j+1] - h_ave[ix+1, j-1]) \
                           - f_ave[ix-1, j  ] * (h_ave[ix-1, j+1] - h_ave[ix-1, j-1]) \
                           - f_ave[ix,   j+1] * (h_ave[ix+1, j+1] - h_ave[ix-1, j+1]) \
                           + f_ave[ix,   j-1] * (h_ave[ix+1, j-1] - h_ave[ix-1, j-1])
                    
                    jcp_J1 = f_ave[ix+1, j+1] * (h_ave[ix,   j+1] - h_ave[ix+1, j  ]) \
                           - f_ave[ix-1, j-1] * (h_ave[ix-1, j  ] - h_ave[ix,   j-1]) \
                           - f_ave[ix-1, j+1] * (h_ave[ix,   j+1] - h_ave[ix-1, j  ]) \
                           + f_ave[ix+1, j-1] * (h_ave[ix+1, j  ] - h_ave[ix,   j-1])
                    
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
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    
                    y[iy, j] = (fp[ix, j] - fh[ix, j]) * self.ht_inv \
                             + result_J4 * self.hx_inv * self.hv_inv \
                             + self.ht * self.regularisation * self.hx2_inv * ( 2. * fp[ix, j] - fp[ix+1, j] - fp[ix-1, j] ) \
                             + self.ht * self.regularisation * self.hv2_inv * ( 2. * fp[ix, j] - fp[ix, j+1] - fp[ix, j-1] )
