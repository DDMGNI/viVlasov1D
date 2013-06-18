'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport Mat, Vec

from vlasov.VIDA    cimport VIDA


cdef class PETScPoissonMatrixFD(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t hx2
    cdef np.float64_t hx2_inv
    
    cdef np.float64_t charge
    
    
    cdef VIDA dax
    cdef VIDA da1
    
    cdef Vec localX
    cdef Vec localN
