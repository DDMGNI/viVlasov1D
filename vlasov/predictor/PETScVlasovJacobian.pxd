'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa cimport PETScArakawa


cdef class PETScVlasovJacobian(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    
    cdef DA da1
    cdef DA dax
    
    cdef Vec B
    cdef Vec X
    
    cdef Vec Fh
    cdef Vec Fp
    
    cdef Vec H0
    cdef Vec H1
    cdef Vec H1h
    cdef Vec H1p
    
    cdef Vec localB
    cdef Vec localX
    
    cdef Vec localF
    cdef Vec localFp
    cdef Vec localFh
    
    cdef Vec localH0
    cdef Vec localH1p
    cdef Vec localH1h
    
    cdef PETScArakawa arakawa


    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
                                            np.uint64_t i, np.uint64_t j)
