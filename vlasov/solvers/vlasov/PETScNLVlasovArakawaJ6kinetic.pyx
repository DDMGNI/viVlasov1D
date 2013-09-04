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
    def jacobian(self, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy
        cdef npy.int64_t xe, xs
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t bracket, coll_drag, coll_diff
        
        self.get_data_arrays()
        
        (xs, xe), = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fd    = self.fd
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v     = self.v
        cdef npy.ndarray[npy.float64_t, ndim=1] u     = self.up
        cdef npy.ndarray[npy.float64_t, ndim=1] a     = self.ap
        
        
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(0, self.nv):
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fd[ix,j]
                    
                else:
                    bracket = 46. * ( \
                                     - fd[ix+1, j  ] * h_ave[ix+1, j+1] \
                                     + fd[ix,   j+1] * h_ave[ix+1, j+1] \
                                     + fd[ix+1, j  ] * h_ave[ix+1, j-1] \
                                     - fd[ix,   j-1] * h_ave[ix+1, j-1] \
                                     + fd[ix+1, j+1] * h_ave[ix+1, j  ] \
                                     - fd[ix+1, j-1] * h_ave[ix+1, j  ] \
                                     + fd[ix,   j+1] * h_ave[ix+1, j  ] \
                                     - fd[ix,   j-1] * h_ave[ix+1, j  ] \
                                     + fd[ix-1, j  ] * h_ave[ix-1, j+1] \
                                     - fd[ix,   j+1] * h_ave[ix-1, j+1] \
                                     - fd[ix-1, j  ] * h_ave[ix-1, j-1] \
                                     + fd[ix,   j-1] * h_ave[ix-1, j-1] \
                                     - fd[ix-1, j+1] * h_ave[ix-1, j  ] \
                                     + fd[ix-1, j-1] * h_ave[ix-1, j  ] \
                                     - fd[ix,   j+1] * h_ave[ix-1, j  ] \
                                     + fd[ix,   j-1] * h_ave[ix-1, j  ] \
                                     - fd[ix+1, j+1] * h_ave[ix,   j+1] \
                                     - fd[ix+1, j  ] * h_ave[ix,   j+1] \
                                     + fd[ix-1, j+1] * h_ave[ix,   j+1] \
                                     + fd[ix-1, j  ] * h_ave[ix,   j+1] \
                                     + fd[ix+1, j-1] * h_ave[ix,   j-1] \
                                     + fd[ix+1, j  ] * h_ave[ix,   j-1] \
                                     - fd[ix-1, j-1] * h_ave[ix,   j-1] \
                                     - fd[ix-1, j  ] * h_ave[ix,   j-1] \
                            )
                    
                    bracket += 18. * ( \
                                     - fd[ix+1, j+1] * h_ave[ix+2, j  ] \
                                     + fd[ix+1, j-1] * h_ave[ix+2, j  ] \
                                     + fd[ix+2, j  ] * h_ave[ix+1, j+1] \
                                     + fd[ix+1, j-1] * h_ave[ix+1, j+1] \
                                     - fd[ix-1, j+1] * h_ave[ix+1, j+1] \
                                     - fd[ix,   j+2] * h_ave[ix+1, j+1] \
                                     - fd[ix+2, j  ] * h_ave[ix+1, j-1] \
                                     - fd[ix+1, j+1] * h_ave[ix+1, j-1] \
                                     + fd[ix-1, j-1] * h_ave[ix+1, j-1] \
                                     + fd[ix,   j-2] * h_ave[ix+1, j-1] \
                                     + fd[ix+1, j+1] * h_ave[ix-1, j+1] \
                                     - fd[ix-1, j-1] * h_ave[ix-1, j+1] \
                                     - fd[ix-2, j  ] * h_ave[ix-1, j+1] \
                                     + fd[ix,   j+2] * h_ave[ix-1, j+1] \
                                     - fd[ix+1, j-1] * h_ave[ix-1, j-1] \
                                     + fd[ix-1, j+1] * h_ave[ix-1, j-1] \
                                     + fd[ix-2, j  ] * h_ave[ix-1, j-1] \
                                     - fd[ix,   j-2] * h_ave[ix-1, j-1] \
                                     + fd[ix-1, j+1] * h_ave[ix-2, j  ] \
                                     - fd[ix-1, j-1] * h_ave[ix-2, j  ] \
                                     + fd[ix+1, j+1] * h_ave[ix,   j+2] \
                                     - fd[ix-1, j+1] * h_ave[ix,   j+2] \
                                     - fd[ix+1, j-1] * h_ave[ix,   j-2] \
                                     + fd[ix-1, j-1] * h_ave[ix,   j-2] \
                             )
                    
                    bracket += \
                             + fd[ix+2, j+1] * h_ave[ix+3, j  ] \
                             - fd[ix+2, j-1] * h_ave[ix+3, j  ] \
                             + fd[ix+1, j+1] * h_ave[ix+3, j  ] \
                             - fd[ix+1, j-1] * h_ave[ix+3, j  ] \
                             - fd[ix+3, j  ] * h_ave[ix+2, j+1] \
                             - fd[ix+1, j+1] * h_ave[ix+2, j+1] \
                             + fd[ix+1, j  ] * h_ave[ix+2, j+1] \
                             + fd[ix-1, j+1] * h_ave[ix+2, j+1] \
                             + fd[ix+3, j  ] * h_ave[ix+2, j-1] \
                             + fd[ix+1, j-1] * h_ave[ix+2, j-1] \
                             - fd[ix+1, j  ] * h_ave[ix+2, j-1] \
                             - fd[ix-1, j-1] * h_ave[ix+2, j-1] \
                             + fd[ix+1, j+1] * h_ave[ix+1, j+2] \
                             - fd[ix+1, j-1] * h_ave[ix+1, j+2] \
                             + fd[ix,   j+3] * h_ave[ix+1, j+2] \
                             - fd[ix,   j+1] * h_ave[ix+1, j+2] \
                             - fd[ix+3, j  ] * h_ave[ix+1, j+1] \
                             + fd[ix+2, j+1] * h_ave[ix+1, j+1] \
                             - fd[ix+1, j+2] * h_ave[ix+1, j+1] \
                             - fd[ix+1, j-2] * h_ave[ix+1, j+1] \
                             - fd[ix-1, j  ] * h_ave[ix+1, j+1] \
                             + fd[ix-2, j+1] * h_ave[ix+1, j+1] \
                             + fd[ix,   j+3] * h_ave[ix+1, j+1] \
                             + fd[ix,   j-1] * h_ave[ix+1, j+1] \
                             + fd[ix+3, j  ] * h_ave[ix+1, j-1] \
                             - fd[ix+2, j-1] * h_ave[ix+1, j-1] \
                             + fd[ix+1, j+2] * h_ave[ix+1, j-1] \
                             + fd[ix+1, j-2] * h_ave[ix+1, j-1] \
                             + fd[ix-1, j  ] * h_ave[ix+1, j-1] \
                             - fd[ix-2, j-1] * h_ave[ix+1, j-1] \
                             - fd[ix,   j+1] * h_ave[ix+1, j-1] \
                             - fd[ix,   j-3] * h_ave[ix+1, j-1] \
                             + fd[ix+1, j+1] * h_ave[ix+1, j-2] \
                             - fd[ix+1, j-1] * h_ave[ix+1, j-2] \
                             + fd[ix,   j-1] * h_ave[ix+1, j-2] \
                             - fd[ix,   j-3] * h_ave[ix+1, j-2] \
                             - fd[ix+2, j+1] * h_ave[ix+1, j  ] \
                             + fd[ix+2, j-1] * h_ave[ix+1, j  ] \
                             - fd[ix-1, j+1] * h_ave[ix+1, j  ] \
                             + fd[ix-1, j-1] * h_ave[ix+1, j  ] \
                             - fd[ix-1, j+1] * h_ave[ix-1, j+2] \
                             + fd[ix-1, j-1] * h_ave[ix-1, j+2] \
                             - fd[ix,   j+3] * h_ave[ix-1, j+2] \
                             + fd[ix,   j+1] * h_ave[ix-1, j+2] \
                             - fd[ix+2, j+1] * h_ave[ix-1, j+1] \
                             + fd[ix+1, j  ] * h_ave[ix-1, j+1] \
                             + fd[ix-1, j+2] * h_ave[ix-1, j+1] \
                             + fd[ix-1, j-2] * h_ave[ix-1, j+1] \
                             - fd[ix-2, j+1] * h_ave[ix-1, j+1] \
                             + fd[ix-3, j  ] * h_ave[ix-1, j+1] \
                             - fd[ix,   j+3] * h_ave[ix-1, j+1] \
                             - fd[ix,   j-1] * h_ave[ix-1, j+1] \
                             + fd[ix+2, j-1] * h_ave[ix-1, j-1] \
                             - fd[ix+1, j  ] * h_ave[ix-1, j-1] \
                             - fd[ix-1, j+2] * h_ave[ix-1, j-1] \
                             - fd[ix-1, j-2] * h_ave[ix-1, j-1] \
                             + fd[ix-2, j-1] * h_ave[ix-1, j-1] \
                             - fd[ix-3, j  ] * h_ave[ix-1, j-1] \
                             + fd[ix,   j+1] * h_ave[ix-1, j-1] \
                             + fd[ix,   j-3] * h_ave[ix-1, j-1] \
                             - fd[ix-1, j+1] * h_ave[ix-1, j-2] \
                             + fd[ix-1, j-1] * h_ave[ix-1, j-2] \
                             - fd[ix,   j-1] * h_ave[ix-1, j-2] \
                             + fd[ix,   j-3] * h_ave[ix-1, j-2] \
                             + fd[ix+1, j+1] * h_ave[ix-1, j  ] \
                             - fd[ix+1, j-1] * h_ave[ix-1, j  ] \
                             + fd[ix-2, j+1] * h_ave[ix-1, j  ] \
                             - fd[ix-2, j-1] * h_ave[ix-1, j  ] \
                             - fd[ix+1, j+1] * h_ave[ix-2, j+1] \
                             + fd[ix-1, j+1] * h_ave[ix-2, j+1] \
                             - fd[ix-1, j  ] * h_ave[ix-2, j+1] \
                             + fd[ix-3, j  ] * h_ave[ix-2, j+1] \
                             + fd[ix+1, j-1] * h_ave[ix-2, j-1] \
                             - fd[ix-1, j-1] * h_ave[ix-2, j-1] \
                             + fd[ix-1, j  ] * h_ave[ix-2, j-1] \
                             - fd[ix-3, j  ] * h_ave[ix-2, j-1] \
                             - fd[ix-1, j+1] * h_ave[ix-3, j  ] \
                             + fd[ix-1, j-1] * h_ave[ix-3, j  ] \
                             - fd[ix-2, j+1] * h_ave[ix-3, j  ] \
                             + fd[ix-2, j-1] * h_ave[ix-3, j  ] \
                             - fd[ix+1, j+2] * h_ave[ix,   j+3] \
                             - fd[ix+1, j+1] * h_ave[ix,   j+3] \
                             + fd[ix-1, j+2] * h_ave[ix,   j+3] \
                             + fd[ix-1, j+1] * h_ave[ix,   j+3] \
                             + fd[ix+1, j+2] * h_ave[ix,   j+1] \
                             + fd[ix+1, j-1] * h_ave[ix,   j+1] \
                             - fd[ix-1, j+2] * h_ave[ix,   j+1] \
                             - fd[ix-1, j-1] * h_ave[ix,   j+1] \
                             - fd[ix+1, j+1] * h_ave[ix,   j-1] \
                             - fd[ix+1, j-2] * h_ave[ix,   j-1] \
                             + fd[ix-1, j+1] * h_ave[ix,   j-1] \
                             + fd[ix-1, j-2] * h_ave[ix,   j-1] \
                             + fd[ix+1, j-1] * h_ave[ix,   j-3] \
                             + fd[ix+1, j-2] * h_ave[ix,   j-3] \
                             - fd[ix-1, j-1] * h_ave[ix,   j-3] \
                             - fd[ix-1, j-2] * h_ave[ix,   j-3]
                    
                    bracket /= 168.
                    
                    
                    # collision operator
                    coll_drag = ( (v[j+1] - u[ix]) * fd[ix, j+1] - (v[j-1] - u[ix]) * fd[ix, j-1] ) * a[ix]
                    coll_diff = ( fd[ix, j+1] - 2. * fd[ix, j] + fd[ix, j-1] )
                    
         
                    y[iy, j] = fd[ix, j] * self.ht_inv \
                             + 0.5 * bracket * self.hx_inv * self.hv_inv \
                             - 0.5 * self.nu * self.coll_drag * coll_drag * self.hv_inv * 0.5 \
                             - 0.5 * self.nu * self.coll_diff * coll_diff * self.hv2_inv \
                             + self.ht * self.regularisation * self.hx2_inv * ( 2. * fd[ix, j] - fd[ix+1, j] - fd[ix-1, j] ) \
                             + self.ht * self.regularisation * self.hv2_inv * ( 2. * fd[ix, j] - fd[ix, j+1] - fd[ix, j-1] )
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy
        cdef npy.int64_t xe, xs
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t bracket, coll_drag, coll_diff
        
        self.get_data_arrays()
        
        (xs, xe), = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fp    = self.fd
        cdef npy.ndarray[npy.float64_t, ndim=2] fh    = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v     = self.v
        cdef npy.ndarray[npy.float64_t, ndim=1] up    = self.up
        cdef npy.ndarray[npy.float64_t, ndim=1] ap    = self.ap
        cdef npy.ndarray[npy.float64_t, ndim=1] uh    = self.uh
        cdef npy.ndarray[npy.float64_t, ndim=1] ah    = self.ah
        
        
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(0, self.nv):
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
                    # Dirichlet Boundary Conditions
                    y[iy, j] = fp[ix,j]
                    
                else:
                    bracket = 46. * ( \
                                     - f_ave[ix+1, j  ] * h_ave[ix+1, j+1] \
                                     + f_ave[ix,   j+1] * h_ave[ix+1, j+1] \
                                     + f_ave[ix+1, j  ] * h_ave[ix+1, j-1] \
                                     - f_ave[ix,   j-1] * h_ave[ix+1, j-1] \
                                     + f_ave[ix+1, j+1] * h_ave[ix+1, j  ] \
                                     - f_ave[ix+1, j-1] * h_ave[ix+1, j  ] \
                                     + f_ave[ix,   j+1] * h_ave[ix+1, j  ] \
                                     - f_ave[ix,   j-1] * h_ave[ix+1, j  ] \
                                     + f_ave[ix-1, j  ] * h_ave[ix-1, j+1] \
                                     - f_ave[ix,   j+1] * h_ave[ix-1, j+1] \
                                     - f_ave[ix-1, j  ] * h_ave[ix-1, j-1] \
                                     + f_ave[ix,   j-1] * h_ave[ix-1, j-1] \
                                     - f_ave[ix-1, j+1] * h_ave[ix-1, j  ] \
                                     + f_ave[ix-1, j-1] * h_ave[ix-1, j  ] \
                                     - f_ave[ix,   j+1] * h_ave[ix-1, j  ] \
                                     + f_ave[ix,   j-1] * h_ave[ix-1, j  ] \
                                     - f_ave[ix+1, j+1] * h_ave[ix,   j+1] \
                                     - f_ave[ix+1, j  ] * h_ave[ix,   j+1] \
                                     + f_ave[ix-1, j+1] * h_ave[ix,   j+1] \
                                     + f_ave[ix-1, j  ] * h_ave[ix,   j+1] \
                                     + f_ave[ix+1, j-1] * h_ave[ix,   j-1] \
                                     + f_ave[ix+1, j  ] * h_ave[ix,   j-1] \
                                     - f_ave[ix-1, j-1] * h_ave[ix,   j-1] \
                                     - f_ave[ix-1, j  ] * h_ave[ix,   j-1] \
                            )
                    
                    bracket += 18. * ( \
                                     - f_ave[ix+1, j+1] * h_ave[ix+2, j  ] \
                                     + f_ave[ix+1, j-1] * h_ave[ix+2, j  ] \
                                     + f_ave[ix+2, j  ] * h_ave[ix+1, j+1] \
                                     + f_ave[ix+1, j-1] * h_ave[ix+1, j+1] \
                                     - f_ave[ix-1, j+1] * h_ave[ix+1, j+1] \
                                     - f_ave[ix,   j+2] * h_ave[ix+1, j+1] \
                                     - f_ave[ix+2, j  ] * h_ave[ix+1, j-1] \
                                     - f_ave[ix+1, j+1] * h_ave[ix+1, j-1] \
                                     + f_ave[ix-1, j-1] * h_ave[ix+1, j-1] \
                                     + f_ave[ix,   j-2] * h_ave[ix+1, j-1] \
                                     + f_ave[ix+1, j+1] * h_ave[ix-1, j+1] \
                                     - f_ave[ix-1, j-1] * h_ave[ix-1, j+1] \
                                     - f_ave[ix-2, j  ] * h_ave[ix-1, j+1] \
                                     + f_ave[ix,   j+2] * h_ave[ix-1, j+1] \
                                     - f_ave[ix+1, j-1] * h_ave[ix-1, j-1] \
                                     + f_ave[ix-1, j+1] * h_ave[ix-1, j-1] \
                                     + f_ave[ix-2, j  ] * h_ave[ix-1, j-1] \
                                     - f_ave[ix,   j-2] * h_ave[ix-1, j-1] \
                                     + f_ave[ix-1, j+1] * h_ave[ix-2, j  ] \
                                     - f_ave[ix-1, j-1] * h_ave[ix-2, j  ] \
                                     + f_ave[ix+1, j+1] * h_ave[ix,   j+2] \
                                     - f_ave[ix-1, j+1] * h_ave[ix,   j+2] \
                                     - f_ave[ix+1, j-1] * h_ave[ix,   j-2] \
                                     + f_ave[ix-1, j-1] * h_ave[ix,   j-2] \
                             )
                    
                    bracket += \
                             + f_ave[ix+2, j+1] * h_ave[ix+3, j  ] \
                             - f_ave[ix+2, j-1] * h_ave[ix+3, j  ] \
                             + f_ave[ix+1, j+1] * h_ave[ix+3, j  ] \
                             - f_ave[ix+1, j-1] * h_ave[ix+3, j  ] \
                             - f_ave[ix+3, j  ] * h_ave[ix+2, j+1] \
                             - f_ave[ix+1, j+1] * h_ave[ix+2, j+1] \
                             + f_ave[ix+1, j  ] * h_ave[ix+2, j+1] \
                             + f_ave[ix-1, j+1] * h_ave[ix+2, j+1] \
                             + f_ave[ix+3, j  ] * h_ave[ix+2, j-1] \
                             + f_ave[ix+1, j-1] * h_ave[ix+2, j-1] \
                             - f_ave[ix+1, j  ] * h_ave[ix+2, j-1] \
                             - f_ave[ix-1, j-1] * h_ave[ix+2, j-1] \
                             + f_ave[ix+1, j+1] * h_ave[ix+1, j+2] \
                             - f_ave[ix+1, j-1] * h_ave[ix+1, j+2] \
                             + f_ave[ix,   j+3] * h_ave[ix+1, j+2] \
                             - f_ave[ix,   j+1] * h_ave[ix+1, j+2] \
                             - f_ave[ix+3, j  ] * h_ave[ix+1, j+1] \
                             + f_ave[ix+2, j+1] * h_ave[ix+1, j+1] \
                             - f_ave[ix+1, j+2] * h_ave[ix+1, j+1] \
                             - f_ave[ix+1, j-2] * h_ave[ix+1, j+1] \
                             - f_ave[ix-1, j  ] * h_ave[ix+1, j+1] \
                             + f_ave[ix-2, j+1] * h_ave[ix+1, j+1] \
                             + f_ave[ix,   j+3] * h_ave[ix+1, j+1] \
                             + f_ave[ix,   j-1] * h_ave[ix+1, j+1] \
                             + f_ave[ix+3, j  ] * h_ave[ix+1, j-1] \
                             - f_ave[ix+2, j-1] * h_ave[ix+1, j-1] \
                             + f_ave[ix+1, j+2] * h_ave[ix+1, j-1] \
                             + f_ave[ix+1, j-2] * h_ave[ix+1, j-1] \
                             + f_ave[ix-1, j  ] * h_ave[ix+1, j-1] \
                             - f_ave[ix-2, j-1] * h_ave[ix+1, j-1] \
                             - f_ave[ix,   j+1] * h_ave[ix+1, j-1] \
                             - f_ave[ix,   j-3] * h_ave[ix+1, j-1] \
                             + f_ave[ix+1, j+1] * h_ave[ix+1, j-2] \
                             - f_ave[ix+1, j-1] * h_ave[ix+1, j-2] \
                             + f_ave[ix,   j-1] * h_ave[ix+1, j-2] \
                             - f_ave[ix,   j-3] * h_ave[ix+1, j-2] \
                             - f_ave[ix+2, j+1] * h_ave[ix+1, j  ] \
                             + f_ave[ix+2, j-1] * h_ave[ix+1, j  ] \
                             - f_ave[ix-1, j+1] * h_ave[ix+1, j  ] \
                             + f_ave[ix-1, j-1] * h_ave[ix+1, j  ] \
                             - f_ave[ix-1, j+1] * h_ave[ix-1, j+2] \
                             + f_ave[ix-1, j-1] * h_ave[ix-1, j+2] \
                             - f_ave[ix,   j+3] * h_ave[ix-1, j+2] \
                             + f_ave[ix,   j+1] * h_ave[ix-1, j+2] \
                             - f_ave[ix+2, j+1] * h_ave[ix-1, j+1] \
                             + f_ave[ix+1, j  ] * h_ave[ix-1, j+1] \
                             + f_ave[ix-1, j+2] * h_ave[ix-1, j+1] \
                             + f_ave[ix-1, j-2] * h_ave[ix-1, j+1] \
                             - f_ave[ix-2, j+1] * h_ave[ix-1, j+1] \
                             + f_ave[ix-3, j  ] * h_ave[ix-1, j+1] \
                             - f_ave[ix,   j+3] * h_ave[ix-1, j+1] \
                             - f_ave[ix,   j-1] * h_ave[ix-1, j+1] \
                             + f_ave[ix+2, j-1] * h_ave[ix-1, j-1] \
                             - f_ave[ix+1, j  ] * h_ave[ix-1, j-1] \
                             - f_ave[ix-1, j+2] * h_ave[ix-1, j-1] \
                             - f_ave[ix-1, j-2] * h_ave[ix-1, j-1] \
                             + f_ave[ix-2, j-1] * h_ave[ix-1, j-1] \
                             - f_ave[ix-3, j  ] * h_ave[ix-1, j-1] \
                             + f_ave[ix,   j+1] * h_ave[ix-1, j-1] \
                             + f_ave[ix,   j-3] * h_ave[ix-1, j-1] \
                             - f_ave[ix-1, j+1] * h_ave[ix-1, j-2] \
                             + f_ave[ix-1, j-1] * h_ave[ix-1, j-2] \
                             - f_ave[ix,   j-1] * h_ave[ix-1, j-2] \
                             + f_ave[ix,   j-3] * h_ave[ix-1, j-2] \
                             + f_ave[ix+1, j+1] * h_ave[ix-1, j  ] \
                             - f_ave[ix+1, j-1] * h_ave[ix-1, j  ] \
                             + f_ave[ix-2, j+1] * h_ave[ix-1, j  ] \
                             - f_ave[ix-2, j-1] * h_ave[ix-1, j  ] \
                             - f_ave[ix+1, j+1] * h_ave[ix-2, j+1] \
                             + f_ave[ix-1, j+1] * h_ave[ix-2, j+1] \
                             - f_ave[ix-1, j  ] * h_ave[ix-2, j+1] \
                             + f_ave[ix-3, j  ] * h_ave[ix-2, j+1] \
                             + f_ave[ix+1, j-1] * h_ave[ix-2, j-1] \
                             - f_ave[ix-1, j-1] * h_ave[ix-2, j-1] \
                             + f_ave[ix-1, j  ] * h_ave[ix-2, j-1] \
                             - f_ave[ix-3, j  ] * h_ave[ix-2, j-1] \
                             - f_ave[ix-1, j+1] * h_ave[ix-3, j  ] \
                             + f_ave[ix-1, j-1] * h_ave[ix-3, j  ] \
                             - f_ave[ix-2, j+1] * h_ave[ix-3, j  ] \
                             + f_ave[ix-2, j-1] * h_ave[ix-3, j  ] \
                             - f_ave[ix+1, j+2] * h_ave[ix,   j+3] \
                             - f_ave[ix+1, j+1] * h_ave[ix,   j+3] \
                             + f_ave[ix-1, j+2] * h_ave[ix,   j+3] \
                             + f_ave[ix-1, j+1] * h_ave[ix,   j+3] \
                             + f_ave[ix+1, j+2] * h_ave[ix,   j+1] \
                             + f_ave[ix+1, j-1] * h_ave[ix,   j+1] \
                             - f_ave[ix-1, j+2] * h_ave[ix,   j+1] \
                             - f_ave[ix-1, j-1] * h_ave[ix,   j+1] \
                             - f_ave[ix+1, j+1] * h_ave[ix,   j-1] \
                             - f_ave[ix+1, j-2] * h_ave[ix,   j-1] \
                             + f_ave[ix-1, j+1] * h_ave[ix,   j-1] \
                             + f_ave[ix-1, j-2] * h_ave[ix,   j-1] \
                             + f_ave[ix+1, j-1] * h_ave[ix,   j-3] \
                             + f_ave[ix+1, j-2] * h_ave[ix,   j-3] \
                             - f_ave[ix-1, j-1] * h_ave[ix,   j-3] \
                             - f_ave[ix-1, j-2] * h_ave[ix,   j-3]
                    
                    bracket /= 168.
                    
                    # collision operator
                    coll_drag = ( (v[j+1] - up[ix]) * fp[ix, j+1] - (v[j-1] - up[ix]) * fp[ix, j-1] ) * ap[ix] \
                              + ( (v[j+1] - uh[ix]) * fh[ix, j+1] - (v[j-1] - uh[ix]) * fh[ix, j-1] ) * ah[ix]
                    coll_diff = ( fp[ix, j+1] - 2. * fp[ix, j] + fp[ix, j-1] ) \
                              + ( fh[ix, j+1] - 2. * fh[ix, j] + fh[ix, j-1] )
                    
                    
                    y[iy, j] = (fp[ix, j] - fh[ix, j]) * self.ht_inv \
                             + 1.0 * bracket * self.hx_inv * self.hv_inv \
                             - 0.5 * self.nu * self.coll_drag * coll_drag * self.hv_inv * 0.5 \
                             - 0.5 * self.nu * self.coll_diff * coll_diff * self.hv2_inv \
                             + self.ht * self.regularisation * self.hx2_inv * ( 2. * fp[ix, j] - fp[ix+1, j] - fp[ix-1, j] ) \
                             + self.ht * self.regularisation * self.hv2_inv * ( 2. * fp[ix, j] - fp[ix, j+1] - fp[ix, j-1] )
