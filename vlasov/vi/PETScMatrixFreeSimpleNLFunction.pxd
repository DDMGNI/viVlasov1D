'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa cimport PETScArakawa


cdef class PETScFunction(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t hx2
    cdef np.float64_t hx2_inv
    
    cdef np.float64_t poisson_const
    
    cdef DA dax
    cdef DA da1
    cdef DA da2
    
    cdef Vec B
    cdef Vec X
    cdef Vec Xh
    cdef Vec Xp
    cdef Vec H0
    cdef Vec F
    cdef Vec PHI
    
    cdef Vec localB
    cdef Vec localX
    cdef Vec localXh
    cdef Vec localXp
    cdef Vec localH0
    
    cdef PETScArakawa arakawa


    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
                                            np.uint64_t i, np.uint64_t j)
