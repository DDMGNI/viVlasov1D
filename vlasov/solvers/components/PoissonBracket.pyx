'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from libc.math cimport exp, pow, sqrt


cdef class PoissonBracket(object):
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
        
        # create local vectors
        self.localF = da1.createLocalVec()
        self.localH = da1.createLocalVec()
        
        
    
    cpdef arakawa_J1(self, Vec F, Vec H, Vec Y, double factor):
        cdef double[:,:] f = self.da1.getLocalArray(F, self.localF)
        cdef double[:,:] h = self.da1.getLocalArray(H, self.localH)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        self.arakawa_J1_array(f, h, y, factor)
        

    cpdef arakawa_J2(self, Vec F, Vec H, Vec Y, double factor):
        cdef double[:,:] f = self.da1.getLocalArray(F, self.localF)
        cdef double[:,:] h = self.da1.getLocalArray(H, self.localH)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        self.arakawa(f, h, y, factor)

    cpdef arakawa_J4(self, Vec F, Vec H, Vec Y, double factor):
        cdef double[:,:] f = self.da1.getLocalArray(F, self.localF)
        cdef double[:,:] h = self.da1.getLocalArray(H, self.localH)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        self.arakawa_J4_array(f, h, y, factor)

    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef arakawa_J1_array(self, double[:,:] x, double[:,:] h, double[:,:] y, double factor):
        
        cdef double jpp, jpc, jcp, result, arakawa_factor
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        arakawa_factor = self.grid.hx_inv * self.grid.hv_inv / 12.
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                # Vlasov equation with Dirichlet Boundary Conditions
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    jpp = (x[ix+1, jx  ] - x[ix-1, jx  ]) * (h[ix,   jx+1] - h[ix,   jx-1]) \
                        - (x[ix,   jx+1] - x[ix,   jx-1]) * (h[ix+1, jx  ] - h[ix-1, jx  ])
                    
                    jpc = x[ix+1, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                        - x[ix-1, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                        - x[ix,   jx+1] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                        + x[ix,   jx-1] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp = x[ix+1, jx+1] * (h[ix,   jx+1] - h[ix+1, jx  ]) \
                        - x[ix-1, jx-1] * (h[ix-1, jx  ] - h[ix,   jx-1]) \
                        - x[ix-1, jx+1] * (h[ix,   jx+1] - h[ix-1, jx  ]) \
                        + x[ix+1, jx-1] * (h[ix+1, jx  ] - h[ix,   jx-1])
        
                    y[iy, jy] += factor * (jpp + jpc + jcp) * arakawa_factor
                    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef arakawa_J2_array(self, double[:,:] x, double[:,:] h, double[:,:] y, double factor):
        
        cdef double jcc, jpc, jcp, result, arakawa_factor
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        arakawa_factor = self.grid.hx_inv * self.grid.hv_inv / 24.
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                # Vlasov equation with Dirichlet Boundary Conditions
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    jcc = (x[ix+1, jx+1] - x[ix-1, jx-1]) * (h[ix-1, jx+1] - h[ix+1, jx-1]) \
                        - (x[ix-1, jx+1] - x[ix+1, jx-1]) * (h[ix+1, jx+1] - h[ix-1, jx-1])
                    
                    jpc = x[ix+2, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                        - x[ix-2, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                        - x[ix,   jx+2] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                        + x[ix,   jx-2] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp = x[ix+1, jx+1] * (h[ix,   jx+2] - h[ix+2, jx  ]) \
                        - x[ix-1, jx-1] * (h[ix-2, jx  ] - h[ix,   jx-2]) \
                        - x[ix-1, jx+1] * (h[ix,   jx+2] - h[ix-2, jx  ]) \
                        + x[ix+1, jx-1] * (h[ix+2, jx  ] - h[ix,   jx-2])
        
                    y[iy, jy] += factor * (jcc + jpc + jcp) * arakawa_factor
                    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef arakawa_J4_array(self, double[:,:] x, double[:,:] h, double[:,:] y, double factor):
        
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double jpp_J1, jpc_J1, jcp_J1
        cdef double jcc_J2, jpc_J2, jcp_J2
        cdef double result_J1, result_J2
        
        cdef arakawa_factor_J1 = self.grid.hx_inv * self.grid.hv_inv / 12.
        cdef arakawa_factor_J2 = self.grid.hx_inv * self.grid.hv_inv / 24.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                # Vlasov equation with Dirichlet Boundary Conditions
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
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
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) * arakawa_factor_J1
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) * arakawa_factor_J2
                    
                    y[iy, jy] += factor * (2. * result_J1 - result_J2)
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double arakawa_J1_point(self, double[:,:] f, double[:,:] h, int i, int j):
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
        
        result = (jpp + jpc + jcp) * self.grid.hx_inv * self.grid.hv_inv / 12.
        
        return result
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double arakawa_J2_point(self, double[:,:] f, double[:,:] h, int i, int j):
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
        
        result = (jcc + jpc + jcp) * self.grid.hx_inv * self.grid.hv_inv / 24.
        
        return result
    
    
    
    cdef double arakawa_J4_point(self, double[:,:] f, double[:,:] h, int i, int j):
        '''
        Arakawa Bracket 4th order
        '''
        
        return 2.0 * self.arakawa_J1(f, h, i, j) - self.arakawa_J2(f, h, i, j)
    
    
