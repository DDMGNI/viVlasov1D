'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from libc.math cimport exp, pow, sqrt


cdef class Arakawa(object):
    '''
    
    '''
    
    def __cinit__(self,
                  VIDA da1  not None,
                  Grid grid not None):
        '''
        Constructor
        '''
        
        # distributed array and grid
        self.da1  = da1
        self.grid = grid
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t arakawa_J1(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.ndarray[np.float64_t, ndim=2] h,
                                       np.uint64_t i, np.uint64_t j):
        '''
        Arakawa Bracket J1 (second order)
        '''
        
        cdef np.float64_t jpp, jpc, jcp, result
        
        jpp = (f[i+1, j  ] - f[i-1, j  ]) * (h[i,   j+1] - h[i,   j-1]) \
            - (f[i,   j+1] - f[i,   j-1]) * (h[i+1, j  ] - h[i-1, j  ])
        
        jpc = f[i+1, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
            - f[i-1, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
            - f[i,   j+1] * (h[i+1, j+1] - h[i-1, j+1]) \
            + f[i,   j-1] * (h[i+1, j-1] - h[i-1, j-1])
        
        jcp = f[i+1, j+1] * (h[i,   j+1] - h[i+1, j  ]) \
            - f[i-1, j-1] * (h[i-1, j  ] - h[i,   j-1]) \
            - f[i-1, j+1] * (h[i,   j+1] - h[i-1, j  ]) \
            + f[i+1, j-1] * (h[i+1, j  ] - h[i,   j-1])
        
        result = (jpp + jpc + jcp) / (12. * self.grid.hx * self.grid.hv)
        
        return result
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t arakawa_J2(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.ndarray[np.float64_t, ndim=2] h,
                                       np.uint64_t i, np.uint64_t j):
        '''
        Arakawa Bracket
        '''
        
        cdef np.float64_t jcc, jpc, jcp, result
        
        jcc = (f[i+1, j+1] - f[i-1, j-1]) * (h[i-1, j+1] - h[i+1, j-1]) \
            - (f[i-1, j+1] - f[i+1, j-1]) * (h[i+1, j+1] - h[i-1, j-1])
        
        jpc = f[i+2, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
            - f[i-2, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
            - f[i,   j+2] * (h[i+1, j+1] - h[i-1, j+1]) \
            + f[i,   j-2] * (h[i+1, j-1] - h[i-1, j-1])
        
        jcp = f[i+1, j+1] * (h[i,   j+2] - h[i+2, j  ]) \
            - f[i-1, j-1] * (h[i-2, j  ] - h[i,   j-2]) \
            - f[i-1, j+1] * (h[i,   j+2] - h[i-2, j  ]) \
            + f[i+1, j-1] * (h[i+2, j  ] - h[i,   j-2])
        
        result = (jcc + jpc + jcp) / (24. * self.grid.hx * self.grid.hv)
        
        return result
    
    
    
    cdef np.float64_t arakawa_J4(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.ndarray[np.float64_t, ndim=2] h,
                                       np.uint64_t i, np.uint64_t j):
        '''
        Arakawa Bracket 4th order
        '''
        
        return 2.0 * self.arakawa_J1(f, h, i, j) - self.arakawa_J2(f, h, i, j)
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef arakawa_J1_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.ndarray[np.float64_t, ndim=2] y,
                                    np.ndarray[np.float64_t, ndim=2] h):
        
        cdef np.float64_t jpp, jpc, jcp, result
        cdef np.uint64_t i, j, ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys
                
                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    # Dirichlet boundary conditions
                    y[iy, jy] = 0.0
                    
                else:
                    # Vlasov equation
                    jpp = (x[i+1, j  ] - x[i-1, j  ]) * (h[i,   j+1] - h[i,   j-1]) \
                        - (x[i,   j+1] - x[i,   j-1]) * (h[i+1, j  ] - h[i-1, j  ])
                    
                    jpc = x[i+1, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
                        - x[i-1, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
                        - x[i,   j+1] * (h[i+1, j+1] - h[i-1, j+1]) \
                        + x[i,   j-1] * (h[i+1, j-1] - h[i-1, j-1])
                    
                    jcp = x[i+1, j+1] * (h[i,   j+1] - h[i+1, j  ]) \
                        - x[i-1, j-1] * (h[i-1, j  ] - h[i,   j-1]) \
                        - x[i-1, j+1] * (h[i,   j+1] - h[i-1, j  ]) \
                        + x[i+1, j-1] * (h[i+1, j  ] - h[i,   j-1])
        
                    y[iy, jy] = - (jpp + jpc + jcp) / (12. * self.grid.hx * self.grid.hv)
                    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef arakawa_J2_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.ndarray[np.float64_t, ndim=2] y,
                                    np.ndarray[np.float64_t, ndim=2] h):
        
        cdef np.float64_t jcc, jpc, jcp, result
        cdef np.uint64_t i, j, ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    # Dirichlet boundary conditions
                    y[iy, jy] = 0.0
                    
                else:
                    # Vlasov equation
                    jcc = (x[i+1, j+1] - x[i-1, j-1]) * (h[i-1, j+1] - h[i+1, j-1]) \
                        - (x[i-1, j+1] - x[i+1, j-1]) * (h[i+1, j+1] - h[i-1, j-1])
                    
                    jpc = x[i+2, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
                        - x[i-2, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
                        - x[i,   j+2] * (h[i+1, j+1] - h[i-1, j+1]) \
                        + x[i,   j-2] * (h[i+1, j-1] - h[i-1, j-1])
                    
                    jcp = x[i+1, j+1] * (h[i,   j+2] - h[i+2, j  ]) \
                        - x[i-1, j-1] * (h[i-2, j  ] - h[i,   j-2]) \
                        - x[i-1, j+1] * (h[i,   j+2] - h[i-2, j  ]) \
                        + x[i+1, j-1] * (h[i+2, j  ] - h[i,   j-2])
        
                    y[iy, jy] = - (jcc + jpc + jcp) / (24. * self.grid.hx * self.grid.hv)
                    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef arakawa_J4_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.ndarray[np.float64_t, ndim=2] y,
                                    np.ndarray[np.float64_t, ndim=2] h):
        
        cdef np.uint64_t i, j, ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        cdef np.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef np.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef np.float64_t result_J1, result_J2, result_J4
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    # Dirichlet boundary conditions
                    y[iy, jy] = 0.0
                    
                else:
                    # Vlasov equation
    
                    jpp_J1 = (x[ix+1, jx  ] - x[ix-1, jx  ]) * (h[ix,   jx+1] - h[ix,   jx-1]) \
                           - (x[ix,   jx+1] - x[ix,   jx-1]) * (h[ix+1, jx  ] - h[ix-1, jx  ])
                    
                    jpc_J1 = x[ix+1, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - x[ix-1, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - x[ix,   jx+1] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + x[ix,   jx-1] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J1 = x[ix+1, jx+1] * (h[ix,   jx+1] - h[ix+1, jx  ]) \
                           - x[ix-1, jx-1] * (h[ix-1, jx  ] - h[ix,   jx-1]) \
                           - x[ix-1, jx+1] * (h[ix,   jx+1] - h[ix-1, jx  ]) \
                           + x[ix+1, jx-1] * (h[ix+1, jx  ] - h[ix,   jx-1])
                    
                    jcc_J2 = (x[ix+1, jx+1] - x[ix-1, jx-1]) * (h[ix-1, jx+1] - h[ix+1, jx-1]) \
                           - (x[ix-1, jx+1] - x[ix+1, jx-1]) * (h[ix+1, jx+1] - h[ix-1, jx-1])
                    
                    jpc_J2 = x[ix+2, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - x[ix-2, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - x[ix,   jx+2] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + x[ix,   jx-2] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J2 = x[ix+1, jx+1] * (h[ix,   jx+2] - h[ix+2, jx  ]) \
                           - x[ix-1, jx-1] * (h[ix-2, jx  ] - h[ix,   jx-2]) \
                           - x[ix-1, jx+1] * (h[ix,   jx+2] - h[ix-2, jx  ]) \
                           + x[ix+1, jx-1] * (h[ix+2, jx  ] - h[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    y[iy, jy] = - result_J4 * self.grid.hx_inv * self.grid.hv_inv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] f,
                                            np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        return f[i, j] * self.grid.ht_inv
    
