'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from libc.math cimport exp, pow, sqrt


cdef class Collisions(object):
    '''
    
    '''
    
    def __cinit__(self,
                  VIDA dax  not None,
                  VIDA da1  not None,
                  Grid grid not None):
        '''
        Constructor
        '''
        
        # distributed arrays and grid
        self.dax  = dax
        self.da1  = da1
        self.grid = grid
        
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t collT1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        return ( (self.v[j+1] - U[i  ]) * f[i,   j+1] - (self.v[j-1] - U[i  ]) * f[i,   j-1] ) * A[i  ] * 0.5 / self.grid.hv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t collT2(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        return ( f[i,   j+1] - 2. * f[i,   j  ] + f[i,   j-1] ) * self.grid.hv2_inv



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t collE1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        return ( (N[i  ] * self.v[j+1] - U[i  ]) * f[i,   j+1] - (N[i  ] * self.v[j-1] - U[i  ]) * f[i,   j-1] ) * A[i  ] * 0.5 / self.grid.hv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t collE2(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        return ( f[i,   j+1] - 2. * f[i,   j  ] + f[i,   j-1] ) * self.grid.hv2_inv



