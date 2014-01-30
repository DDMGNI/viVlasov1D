'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid cimport Grid
from VIDA cimport VIDA

from petsc4py.PETSc cimport Vec


cdef class Collisions(object):

    cdef VIDA dax
    cdef VIDA da1
    cdef Grid grid
    
    
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
    
