'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython
cimport numpy as np

from petsc4py.PETSc cimport DA, Vec


cdef class PETScArakawa(object):
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef DA da1
    
    
    cdef np.float64_t arakawa(self, np.ndarray[np.float64_t, ndim=2] x,
                                    np.ndarray[np.float64_t, ndim=2] h,
                                    np.uint64_t i, np.uint64_t j)
                                         
    cdef arakawa_timestep(self, np.ndarray[np.float64_t, ndim=2] tx,
                                np.ndarray[np.float64_t, ndim=2] ty,
                                np.ndarray[np.float64_t, ndim=2] h0,
                                np.ndarray[np.float64_t, ndim=2] h1)

