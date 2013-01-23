'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa cimport PETScArakawa


cdef class PETScJacobian(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t hx2
    cdef np.float64_t hv2
    cdef np.float64_t hx2_inv
    cdef np.float64_t hv2_inv
    
    cdef np.float64_t poisson_const
    cdef np.float64_t alpha
    
    cdef np.ndarray v
    
    cdef DA dax
    cdef DA da1
    cdef DA da2
    
    cdef Vec H0
    cdef Vec H1p
    cdef Vec H1h
    cdef Vec Fp
    cdef Vec Fh
    
    cdef Vec A1p
    cdef Vec A1h
    cdef Vec A2p
    cdef Vec A2h
    
    cdef Vec localH0
    cdef Vec localH1p
    cdef Vec localH1h
    cdef Vec localFp
    cdef Vec localFh
    
    cdef Vec localA1p
    cdef Vec localA1h
    cdef Vec localA2p
    cdef Vec localA2h
    