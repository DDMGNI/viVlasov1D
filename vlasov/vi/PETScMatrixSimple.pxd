'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa cimport PETScArakawa


cdef class PETScMatrix(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t hx2
    cdef np.float64_t hv2
    cdef np.float64_t hx2_inv
    cdef np.float64_t hv2_inv
    
    cdef np.ndarray v
    
    cdef np.float64_t poisson_const
    cdef np.float64_t alpha
    
    cdef DA dax
    cdef DA day
    cdef DA da1
    cdef DA da2
    
    cdef Vec VF
    cdef Vec H0
    
    cdef Vec localB
    cdef Vec localF
    cdef Vec localFh
    
    cdef Vec localVF
    cdef Vec localH0
    cdef Vec localH1
    
    cdef PETScArakawa arakawa


    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
                                            np.uint64_t i, np.uint64_t j)
