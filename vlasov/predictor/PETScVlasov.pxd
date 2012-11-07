'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa cimport PETScArakawa


cdef class PETScVlasovSolver(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    
    cdef DA da1
    cdef DA da2
    
    cdef Vec B
    cdef Vec X
    
    cdef Vec H0
    cdef Vec X0
    cdef Vec X1
    
    cdef Vec localB
    cdef Vec localX
    
    cdef Vec localH0
    cdef Vec localX0
    cdef Vec localX1
    
    cdef PETScArakawa arakawa


    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
                                            np.uint64_t i, np.uint64_t j)