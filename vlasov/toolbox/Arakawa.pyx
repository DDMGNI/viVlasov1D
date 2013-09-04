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
    
    def __cinit__(self, VIDA da1, VIDA dax, 
                  np.ndarray[np.float64_t, ndim=1] v,
                  np.uint64_t  nx, np.uint64_t  nv,
                  np.float64_t ht, np.float64_t hx, np.float64_t hv):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.dax = dax
        self.da1 = da1
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv

        self.ht_inv  = 1. / self.ht 
        self.hx_inv  = 1. / self.hx 
        self.hv_inv  = 1. / self.hv
         
        self.hx2     = hx**2
        self.hx2_inv = 1. / self.hx2 
        
        self.hv2     = hv**2
        self.hv2_inv = 1. / self.hv2 
        
        # velocity grid
        self.v = v.copy()
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t arakawa_J1(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.ndarray[np.float64_t, ndim=2] h,
                                       np.uint64_t i, np.uint64_t j):
        '''
        Arakawa Bracket
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
        
        result = (jpp + jpc + jcp) / (12. * self.hx * self.hv)
        
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
        
        result = (jcc + jpc + jcp) / (24. * self.hx * self.hv)
        
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
    cdef arakawa_J1_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            for j in range(0, self.nv):
                ix = i-xs+self.da1.getStencilWidth()
                iy = i-xs
                
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = - self.arakawa_J1(x, h, ix, j)
                    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef arakawa_J2_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            for j in range(0, self.nv):
                ix = i-xs+self.da1.getStencilWidth()
                iy = i-xs
                
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = - self.arakawa_J2(x, h, ix, j)
                    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef arakawa_J4_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef np.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef np.float64_t result_J1, result_J2, result_J4
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            for j in range(0, self.nv):
                ix = i-xs+self.da1.getStencilWidth()
                iy = i-xs
                
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
#                     y[iy, j] = - self.arakawa_J4(x, h, ix, j)
    
                    jpp_J1 = (x[ix+1, j  ] - x[ix-1, j  ]) * (h[ix,   j+1] - h[ix,   j-1]) \
                           - (x[ix,   j+1] - x[ix,   j-1]) * (h[ix+1, j  ] - h[ix-1, j  ])
                    
                    jpc_J1 = x[ix+1, j  ] * (h[ix+1, j+1] - h[ix+1, j-1]) \
                           - x[ix-1, j  ] * (h[ix-1, j+1] - h[ix-1, j-1]) \
                           - x[ix,   j+1] * (h[ix+1, j+1] - h[ix-1, j+1]) \
                           + x[ix,   j-1] * (h[ix+1, j-1] - h[ix-1, j-1])
                    
                    jcp_J1 = x[ix+1, j+1] * (h[ix,   j+1] - h[ix+1, j  ]) \
                           - x[ix-1, j-1] * (h[ix-1, j  ] - h[ix,   j-1]) \
                           - x[ix-1, j+1] * (h[ix,   j+1] - h[ix-1, j  ]) \
                           + x[ix+1, j-1] * (h[ix+1, j  ] - h[ix,   j-1])
                    
                    jcc_J2 = (x[ix+1, j+1] - x[ix-1, j-1]) * (h[ix-1, j+1] - h[ix+1, j-1]) \
                           - (x[ix-1, j+1] - x[ix+1, j-1]) * (h[ix+1, j+1] - h[ix-1, j-1])
                    
                    jpc_J2 = x[ix+2, j  ] * (h[ix+1, j+1] - h[ix+1, j-1]) \
                           - x[ix-2, j  ] * (h[ix-1, j+1] - h[ix-1, j-1]) \
                           - x[ix,   j+2] * (h[ix+1, j+1] - h[ix-1, j+1]) \
                           + x[ix,   j-2] * (h[ix+1, j-1] - h[ix-1, j-1])
                    
                    jcp_J2 = x[ix+1, j+1] * (h[ix,   j+2] - h[ix+2, j  ]) \
                           - x[ix-1, j-1] * (h[ix-2, j  ] - h[ix,   j-2]) \
                           - x[ix-1, j+1] * (h[ix,   j+2] - h[ix-2, j  ]) \
                           + x[ix+1, j-1] * (h[ix+2, j  ] - h[ix,   j-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    
                    y[iy, j] = - result_J4 * self.hx_inv * self.hv_inv
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef arakawa_J4_timestep_h(self, np.ndarray[np.float64_t, ndim=2] x,
                                     np.ndarray[np.float64_t, ndim=2] y,
                                     np.ndarray[np.float64_t, ndim=2] h):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef np.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef np.float64_t result_J1, result_J2, result_J4
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            for j in range(0, self.nv):
                ix = i-xs+self.da1.getStencilWidth()
                iy = i-xs
                
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
    
                    jpp_J1 = (x[ix+1, j  ] - x[ix-1, j  ]) * (h[ix,   j+1] - h[ix,   j-1]) \
                           - (x[ix,   j+1] - x[ix,   j-1]) * (h[ix+1, j  ] - h[ix-1, j  ])
                    
                    jpc_J1 = x[ix+1, j  ] * (h[ix+1, j+1] - h[ix+1, j-1]) \
                           - x[ix-1, j  ] * (h[ix-1, j+1] - h[ix-1, j-1]) \
                           - x[ix,   j+1] * (h[ix+1, j+1] - h[ix-1, j+1]) \
                           + x[ix,   j-1] * (h[ix+1, j-1] - h[ix-1, j-1])
                    
                    jcp_J1 = x[ix+1, j+1] * (h[ix,   j+1] - h[ix+1, j  ]) \
                           - x[ix-1, j-1] * (h[ix-1, j  ] - h[ix,   j-1]) \
                           - x[ix-1, j+1] * (h[ix,   j+1] - h[ix-1, j  ]) \
                           + x[ix+1, j-1] * (h[ix+1, j  ] - h[ix,   j-1])
                    
                    jcc_J2 = (x[ix+1, j+1] - x[ix-1, j-1]) * (h[ix-1, j+1] - h[ix+1, j-1]) \
                           - (x[ix-1, j+1] - x[ix+1, j-1]) * (h[ix+1, j+1] - h[ix-1, j-1])
                    
                    jpc_J2 = x[ix+2, j  ] * (h[ix+1, j+1] - h[ix+1, j-1]) \
                           - x[ix-2, j  ] * (h[ix-1, j+1] - h[ix-1, j-1]) \
                           - x[ix,   j+2] * (h[ix+1, j+1] - h[ix-1, j+1]) \
                           + x[ix,   j-2] * (h[ix+1, j-1] - h[ix-1, j-1])
                    
                    jcp_J2 = x[ix+1, j+1] * (h[ix,   j+2] - h[ix+2, j  ]) \
                           - x[ix-1, j-1] * (h[ix-2, j  ] - h[ix,   j-2]) \
                           - x[ix-1, j+1] * (h[ix,   j+2] - h[ix-2, j  ]) \
                           + x[ix+1, j-1] * (h[ix+2, j  ] - h[ix,   j-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    
                    y[iy, j] = - result_J4 * self.hx_inv * self.hv_inv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] f,
                                            np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        return f[i,j] * self.ht_inv
    
