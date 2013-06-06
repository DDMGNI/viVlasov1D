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
    def formJacobian(self, Mat A):
        cdef npy.int64_t i, j, ix
        cdef npy.int64_t xe, xs
        
        (xs, xe), = self.da1.getRanges()
        
        self.get_data_arrays()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (self.fd + self.fh)
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
        
        
        # Vlasov Equation
        for i in npy.arange(xs, xe):
            ix = i-xs+2
            
            row.index = (i,)
#             col.index = (i,)
                
            for j in npy.arange(0, self.nv):
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
                        ]:

                        col.index = index
                        col.field = field
                        A.setValueStencil(row, col, value)
                        
        
        A.assemble()



    @cython.boundscheck(False)
    def jacobian(self, Vec Y):
        cdef npy.uint64_t i, j
        cdef npy.uint64_t ix, iy
        cdef npy.uint64_t xe, xs
        
        self.get_data_arrays()
        
        (xs, xe), = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        for i in npy.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Vlasov equation
            for j in npy.arange(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = self.fd[ix,j]
                    
                else:
                    ### TODO ###
                    ### collision operator not complete ###
                    ### TODO ###
                    y[iy, j] = self.toolbox.time_derivative(self.fd, ix, j) \
                             + 0.5 * self.toolbox.arakawa_J4(self.fd, h_ave,    ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fd, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fd, self.np, self.up, self.ep, self.ap, ix, j)
            
            
    
    @cython.boundscheck(False)
    def function(self, Vec Y):
        cdef npy.uint64_t i, j
        cdef npy.uint64_t ix, iy
        cdef npy.uint64_t xe, xs
        
        self.get_data_arrays()
        
        (xs, xe), = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (self.fd + self.fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        
        for i in npy.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            # Vlasov equation
            for j in npy.arange(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet Boundary Conditions
                    y[iy, j] = self.fd[ix,j]
                    
                else:
                    y[iy, j] = self.toolbox.time_derivative(self.fd, ix, j) \
                             - self.toolbox.time_derivative(self.fh, ix, j) \
                             + self.toolbox.arakawa_J4(f_ave, h_ave, ix, j) #\
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fp, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT1(self.fh, self.nh, self.uh, self.eh, self.ah, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fp, self.np, self.up, self.ep, self.ap, ix, j) \
#                              - 0.5 * self.nu * self.toolbox.collT2(self.fh, self.nh, self.uh, self.eh, self.ah, ix, j)
