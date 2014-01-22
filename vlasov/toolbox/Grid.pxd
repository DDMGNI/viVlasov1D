'''
Created on Jan 16, 2014

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np


cdef class Grid(object):

    cdef readonly np.uint64_t  stencil
    
    cdef readonly np.ndarray x
    cdef readonly np.ndarray v
    cdef readonly np.ndarray v2
    
    cdef readonly np.uint64_t  nt
    cdef readonly np.uint64_t  nx
    cdef readonly np.uint64_t  nv
    
    cdef readonly np.float64_t ht
    cdef readonly np.float64_t hx
    cdef readonly np.float64_t hv
    
    cdef readonly np.float64_t ht_inv
    cdef readonly np.float64_t hx_inv
    cdef readonly np.float64_t hv_inv
    
    cdef readonly np.float64_t hx2
    cdef readonly np.float64_t hv2
    
    cdef readonly np.float64_t hx2_inv
    cdef readonly np.float64_t hv2_inv
    
