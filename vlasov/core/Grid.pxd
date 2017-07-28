'''
Created on Jan 16, 2014

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np


cdef class Grid(object):

    cdef readonly int stencil
    
    cdef readonly double[:] t
    cdef readonly double[:] x
    cdef readonly double[:] v
    
    cdef readonly double[:] v2
    
    cdef readonly int    nt
    cdef readonly int    nx
    cdef readonly int    nv
    
    cdef readonly double ht
    cdef readonly double hx
    cdef readonly double hv
    
    cdef readonly double ht_inv
    cdef readonly double hx_inv
    cdef readonly double hv_inv
    
    cdef readonly double hx2
    cdef readonly double hv2
    
    cdef readonly double hx2_inv
    cdef readonly double hv2_inv
    
#     cdef readonly np.float64_t tMin
#     cdef readonly np.float64_t tMax
#     cdef readonly np.float64_t xMin
#     cdef readonly np.float64_t xMax
#     cdef readonly np.float64_t vMin
#     cdef readonly np.float64_t vMax
