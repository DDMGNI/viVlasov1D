'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython
cimport numpy as np

from petsc4py.PETSc cimport DA, Vec

from vlasov.predictor.PETScArakawa cimport PETScArakawa


cdef class PETScArakawaRK4(object):
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef DA da1
    cdef DA da2
    
    cdef Vec H0
    
    cdef Vec X1
    cdef Vec X2
    cdef Vec X3
    cdef Vec X4
    
    cdef Vec localX
    cdef Vec localX1
    cdef Vec localX2
    cdef Vec localX3
    cdef Vec localX4
    cdef Vec localH0
    
    cdef PETScArakawa arakawa
    
    
#    cpdef rk4(self, Vec X, np.ndarray[np.float64_t, ndim=1] h0)
