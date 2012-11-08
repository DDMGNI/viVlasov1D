'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa cimport PETScArakawa


cdef class PETScSolver(object):

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
    
    cdef Vec B
    cdef Vec X
    cdef Vec X1
    
    cdef Vec N
    cdef Vec U
    cdef Vec E
    
    cdef Vec A1
    cdef Vec A1h
    cdef Vec A2
    cdef Vec A2h
    
    cdef Vec H0
#    cdef Vec V
    cdef Vec F
    cdef Vec PHI
    
    cdef Vec localB
    cdef Vec localX
    cdef Vec localX1
    
    cdef Vec localA1
    cdef Vec localA1h
    cdef Vec localA2
    cdef Vec localA2h
    
    cdef Vec localH0
    
    cdef PETScArakawa arakawa


    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
                                            np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t dvdv(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t C1(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.ndarray[np.float64_t, ndim=1] A,
                               np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t C2(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.ndarray[np.float64_t, ndim=1] A,
                               np.ndarray[np.float64_t, ndim=1] V,
                               np.uint64_t i, np.uint64_t j)
    
