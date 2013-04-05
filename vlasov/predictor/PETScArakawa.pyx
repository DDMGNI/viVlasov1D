'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport DA, Vec


cdef class PETScArakawa(object):
    '''
    Cython Implementation of Arakawa Discretisation
    '''
    
    
    def __cinit__(self, DA da1,
                  np.uint64_t  nx, np.uint64_t  nv,
                  np.float64_t hx, np.float64_t hv):
        '''
        Constructor
        '''
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.hx = hx
        self.hv = hv

        # disstributed array
        self.da1 = da1
        
    
    @cython.boundscheck(False)
    cdef np.float64_t arakawa(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.ndarray[np.float64_t, ndim=2] h,
                                    np.uint64_t i, np.uint64_t j):
        '''
        Arakawa Bracket
        '''
        
        cdef np.float64_t jpp, jpc, jcp, result
        
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
        
        result = (jpp + jpc + jcp) / (12. * self.hx * self.hv)
        
        return result
    
    
    @cython.boundscheck(False)
    cdef arakawa_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                np.ndarray[np.float64_t, ndim=2] y,
                                np.ndarray[np.float64_t, ndim=2] h0,
                                np.ndarray[np.float64_t, ndim=2] h1):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in np.arange(xs, xe):
            for j in np.arange(0, self.nv):
                ix = i-xs+1
                iy = i-xs
                
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = - self.arakawa(x, h, ix, j)
                    
    
