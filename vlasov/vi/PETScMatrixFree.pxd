'''
Created on Nov 14, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa cimport PETScArakawa


cdef class PETScSolver(object):

    cdef np.uint64_t nx
    cdef np.uint64_t nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t eps
    cdef np.float64_t poisson_const
    
    cdef DA da
    
    cdef Vec B
    cdef Vec X
    cdef Vec X1
    cdef Vec X2
    
    cdef Vec localB
    cdef Vec localX
    cdef Vec localX1
    cdef Vec localX2
    
    cdef np.ndarray h0
    cdef np.ndarray ty
    
    
