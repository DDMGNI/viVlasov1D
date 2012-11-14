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
    
    cdef np.float64_t hx2
    cdef np.float64_t hv2
    cdef np.float64_t hx2_inv
    cdef np.float64_t hv2_inv
    
    cdef np.float64_t alpha
    
    cdef np.ndarray v
    
    
    cdef DA da1
    
    cdef Vec Fh
    cdef Vec VF
    cdef Vec VFh
    
    cdef Vec H0
    cdef Vec H1
    cdef Vec H1h
    
    cdef Vec localB
    cdef Vec localF
    cdef Vec localFh
    cdef Vec localVF
    cdef Vec localVFh
    
    cdef Vec localH0
    cdef Vec localH1
    cdef Vec localH1h
    
    cdef PETScArakawa arakawa


    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
                                            np.uint64_t i, np.uint64_t j)

    cdef np.float64_t dvdv(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t coll(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j)
