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
    
    
    
    cdef np.float64_t arakawa_J1(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.ndarray[np.float64_t, ndim=2] h,
                                       np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t arakawa_J2(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.ndarray[np.float64_t, ndim=2] h,
                                       np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t arakawa_J4(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.ndarray[np.float64_t, ndim=2] h,
                                       np.uint64_t i, np.uint64_t j)
    
    
    cdef arakawa_J1_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1)
    
    cdef arakawa_J2_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1)

    cdef arakawa_J4_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1)
    
    
    cdef np.float64_t average_J1(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t average_J2(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t average_J4(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.uint64_t i, np.uint64_t j)
    
    
    cdef np.float64_t time_derivative_woa(self, np.ndarray[np.float64_t, ndim=2] f,
                                                np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t time_derivative_J1(self, np.ndarray[np.float64_t, ndim=2] f,
                                               np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t time_derivative_J2(self, np.ndarray[np.float64_t, ndim=2] f,
                                               np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t time_derivative_J4(self, np.ndarray[np.float64_t, ndim=2] f,
                                               np.uint64_t i, np.uint64_t j)
    
    
    cdef np.float64_t collT1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collT2(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j)
    
    
    cdef np.float64_t collT1woa(self, np.ndarray[np.float64_t, ndim=2] f,
                                      np.ndarray[np.float64_t, ndim=1] N,
                                      np.ndarray[np.float64_t, ndim=1] U,
                                      np.ndarray[np.float64_t, ndim=1] E,
                                      np.ndarray[np.float64_t, ndim=1] A,
                                      np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collT2woa(self, np.ndarray[np.float64_t, ndim=2] f,
                                      np.ndarray[np.float64_t, ndim=1] N,
                                      np.ndarray[np.float64_t, ndim=1] U,
                                      np.ndarray[np.float64_t, ndim=1] E,
                                      np.ndarray[np.float64_t, ndim=1] A,
                                      np.uint64_t i, np.uint64_t j)
    
    
    cdef np.float64_t collE1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collE2(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collD4(self, np.ndarray[np.float64_t, ndim=2] f,
                               np.uint64_t i, np.uint64_t j)
