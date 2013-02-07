'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from petsc4py.PETSc cimport DA, Vec


cdef class Toolbox(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t ht_inv
    cdef np.float64_t hx_inv
    cdef np.float64_t hv_inv
    
    cdef np.float64_t hx2
    cdef np.float64_t hv2
    cdef np.float64_t hx2_inv
    cdef np.float64_t hv2_inv
    
    cdef np.ndarray v
    
    cdef DA dax
    cdef DA da1
    cdef DA da2
    
    cdef Vec localF
    
    
    
    cdef np.float64_t arakawa(self, np.ndarray[np.float64_t, ndim=2] f,
                                    np.ndarray[np.float64_t, ndim=2] h,
                                    np.uint64_t i, np.uint64_t j)
                                         
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] f,
                                            np.uint64_t i, np.uint64_t j)
    
    
    cdef np.float64_t collT1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] A1,
                                   np.ndarray[np.float64_t, ndim=1] A2,
                                   np.ndarray[np.float64_t, ndim=1] A3,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collT2(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collT_moments(self, Vec F, Vec A1, Vec A2, Vec A3, Vec N, Vec U, Vec E)
    
    
    cdef np.float64_t collE1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] A1,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collE2(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] A2,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collE_moments(self, Vec F, Vec A1, Vec A2)


    cdef np.float64_t collN1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] A1,
                                   np.ndarray[np.float64_t, ndim=1] A2,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collN2(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] A3,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collN_moments(self, Vec F, Vec A1, Vec A2, Vec A3, Vec N, Vec U, Vec E)
