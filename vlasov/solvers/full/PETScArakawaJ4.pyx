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
        cdef npy.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        self.get_data_arrays()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fh = self.fh
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
        
        
        # Poisson equation
        for i in range(xs, xe):
            row.index = (i,)
            row.field = self.grid.nv
            
            # charge density
            col.index = (i,  )
            col.field = self.grid.nv+1
            A.setValueStencil(row, col, self.charge)
            
            
            # Laplace operator
            for index, value in [
                    ((i-1,), - 1. * self.grid.hx2_inv),
                    ((i,  ), + 2. * self.grid.hx2_inv),
                    ((i+1,), - 1. * self.grid.hx2_inv),
                ]:
                
                col.index = index
                col.field = self.grid.nv
                A.setValueStencil(row, col, value)
         
            
        
        # moments
        for i in range(xs, xe):
            ix = i-xs+2
            
            row.index = (i,)
            col.index = (i,)
            
            
            # density
            row.field = self.grid.nv+1
            col.field = self.grid.nv+1
            
            A.setValueStencil(row, col, 1.)
            
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                col.field = j
                A.setValueStencil(row, col, - 1. * self.grid.hv)
             
            
            # average velocity density
            row.field = self.grid.nv+2
            col.field = self.grid.nv+2
            
            A.setValueStencil(row, col, 1.)
            
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                col.field = j
                A.setValueStencil(row, col, - self.v[j] * self.grid.hv)
            
            
            # average energy density
            row.field = self.grid.nv+3
            col.field = self.grid.nv+3
            
            A.setValueStencil(row, col, 1.)
            
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                col.field = j
                A.setValueStencil(row, col, - self.v[j]**2 * self.grid.hv)
                
            
        
        # Vlasov Equation
        for i in range(xs, xe):
            ix = i-xs+2
            
            row.index = (i,)
#             col.index = (i,)
                
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
                            
                            ((i-2,), self.grid.nv,    + 1. * (fh[ix-1, jx+1] - fh[ix-1, jx-1]) * arak_fac_J2),
                            ((i-1,), self.grid.nv,    + 2. * (fh[ix,   jx+1] - fh[ix,   jx-1]) * arak_fac_J1 \
                                                 + 1. * (fh[ix-1, jx+1] - fh[ix-1, jx-1]) * arak_fac_J1 \
                                                 + 1. * (fh[ix-1, jx+1] - fh[ix+1, jx-1]) * arak_fac_J2 \
                                                 + 1. * (fh[ix+1, jx+1] - fh[ix-1, jx-1]) * arak_fac_J2 \
                                                 + 1. * (fh[ix-2, jx  ] - fh[ix,   jx-2]) * arak_fac_J2 \
                                                 + 1. * (fh[ix,   jx+2] - fh[ix-2, jx  ]) * arak_fac_J2),
                            ((i,  ), self.grid.nv,    - 1. * (fh[ix+1, jx-1] - fh[ix-1, jx-1]) * arak_fac_J1 \
                                                 + 1. * (fh[ix+1, jx+1] - fh[ix-1, jx+1]) * arak_fac_J1 \
                                                 + 1. * (fh[ix+1, jx+1] - fh[ix-1, jx+1]) * arak_fac_J2 \
                                                 - 1. * (fh[ix+1, jx-1] - fh[ix-1, jx-1]) * arak_fac_J2),
                            ((i+1,), self.grid.nv,    - 2. * (fh[ix,   jx+1] - fh[ix,   jx-1]) * arak_fac_J1 \
                                                 - 1. * (fh[ix+1, jx+1] - fh[ix+1, jx-1]) * arak_fac_J1 \
                                                 - 1. * (fh[ix-1, jx+1] - fh[ix+1, jx-1]) * arak_fac_J2 \
                                                 - 1. * (fh[ix+1, jx+1] - fh[ix-1, jx-1]) * arak_fac_J2 \
                                                 - 1. * (fh[ix,   jx+2] - fh[ix+2, jx  ]) * arak_fac_J2 \
                                                 - 1. * (fh[ix+2, jx  ] - fh[ix,   jx-2]) * arak_fac_J2),
                            ((i+2,), self.grid.nv,    - 1. * (fh[ix+1, jx+1] - fh[ix+1, jx-1]) * arak_fac_J2),
                            
                            
#                             ### TODO ###
#                             check the following lines
#                             not updodate anymore
#                             ### TODO ###
#                             ((i,  ), self.grid.nv+1,  + coll1_fac * self.fp[ix,   jx+1] * self.v[jx+1] * self.ap[ix  ] \
#                                                  - coll1_fac * self.fp[ix,   jx-1] * self.v[jx-1] * self.ap[ix  ] \
#                                                  + coll1_fac * self.fp[ix,   jx+1] * ( self.np[ix  ] * self.v[jx+1] - self.up[ix  ] ) * ( self.ap[ix  ] / self.np[ix  ] - self.ep[ix  ] * self.ap[ix  ]**2 / self.np[ix  ] ) \
#                                                  - coll1_fac * self.fp[ix,   jx-1] * ( self.np[ix  ] * self.v[jx-1] - self.up[ix  ] ) * ( self.ap[ix  ] / self.np[ix  ] - self.ep[ix  ] * self.ap[ix  ]**2 / self.np[ix  ] ) ),
#                              
#                             ((i,  ), self.grid.nv+2,  - coll1_fac * self.fp[ix,   jx+1] * self.ap[ix  ] \
#                                                  + coll1_fac * self.fp[ix,   jx-1] * self.ap[ix  ] \
#                                                  + coll1_fac * self.fp[ix,   jx+1] * ( self.np[ix  ] * self.v[jx+1] - self.up[ix  ] ) * 2. * self.up[ix  ] * self.ap[ix  ]**2 / self.np[ix  ] \
#                                                  - coll1_fac * self.fp[ix,   jx-1] * ( self.np[ix  ] * self.v[jx-1] - self.up[ix  ] ) * 2. * self.up[ix  ] * self.ap[ix  ]**2 / self.np[ix  ] ),
#                              
#                             ((i,  ), self.grid.nv+3,  - coll1_fac * self.fp[ix,   jx+1] * ( self.np[ix  ] * self.v[jx+1] - self.up[ix  ] ) * self.ap[ix  ]**2 \
#                                                  + coll1_fac * self.fp[ix,   jx-1] * ( self.np[ix  ] * self.v[jx-1] - self.up[ix  ] ) * self.ap[ix  ]**2 ),
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
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da2.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fd = self.fd
        cdef npy.ndarray[npy.float64_t, ndim=2] fh = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] hd = self.h1d
        cdef npy.ndarray[npy.float64_t, ndim=2] hh = self.h0 + self.h1h + self.h2h
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Poisson equation
            y[iy, self.grid.nv] = - ( self.pd[ix-1] + self.pd[ix+1] - 2. * self.pd[ix] ) * self.grid.hx2_inv + self.charge * self.nd[ix]
            
            
            # moments
            y[iy, self.grid.nv+1] = self.nd[ix] - self.nc[ix]
            y[iy, self.grid.nv+2] = self.ud[ix] - self.uc[ix]
            y[iy, self.grid.nv+3] = self.ed[ix] - self.ec[ix]
            
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j <= 1 or j >= self.grid.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = self.fd[ix, jx]
                    
                else:
                    ### TODO ###
                    ### collision operator not complete ###
                    ### TODO ###
                    y[iy, jy] = self.toolbox.time_derivative(fd, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(fd, hh, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(fh, hd, ix, j) \
                             + self.grid.ht * self.regularisation * self.grid.hx2_inv * ( 2. * fd[ix, jx] - fd[ix+1, jx] - fd[ix-1, jx] ) \
                             + self.grid.ht * self.regularisation * self.grid.hv2_inv * ( 2. * fd[ix, jx] - fd[ix, jx+1] - fd[ix, jx-1] ) #\
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
        
        cdef npy.float64_t nmean = self.Nd.sum() / self.grid.nx
        
        self.toolbox.compute_density(self.Fd, self.Nc)
        self.toolbox.compute_velocity_density(self.Fd, self.Uc)
        self.toolbox.compute_energy_density(self.Fd, self.Ec)
        
        self.get_data_arrays()
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da2.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fp = self.fd
        cdef npy.ndarray[npy.float64_t, ndim=2] fh = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] hp = self.h0 + self.h1d + self.h2p
        cdef npy.ndarray[npy.float64_t, ndim=2] hh = self.h0 + self.h1h + self.h2h
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Poisson equation
            y[iy, self.grid.nv] = - ( self.pd[ix-1] + self.pd[ix+1] - 2. * self.pd[ix] ) * self.grid.hx2_inv + self.charge * (self.nd[ix] - nmean)
            
            
            # moments
            y[iy, self.grid.nv+1] = self.nd[ix] - self.nc[ix]
            y[iy, self.grid.nv+2] = self.ud[ix] - self.uc[ix]
            y[iy, self.grid.nv+3] = self.ed[ix] - self.ec[ix]
            
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j <= 1 or j >= self.grid.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = self.fp[ix, jx]
                    
                else:
                    y[iy, jy] = self.toolbox.time_derivative(self.fd, ix, j) \
                             - self.toolbox.time_derivative(self.fh, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(fp, hh, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(fh, hp, ix, j) #\
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fp, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fh, self.nh, self.uh, self.eh, self.ah, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fp, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fh, self.nh, self.uh, self.eh, self.ah, ix, j)
