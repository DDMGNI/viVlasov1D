'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.vi.Toolbox cimport Toolbox


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
    
    cdef np.float64_t charge
    cdef np.float64_t nu
    
    cdef np.ndarray v
    
    cdef DA dax
    cdef DA da1
    cdef DA da2
    
    cdef Vec H0
    cdef Vec H1p
    cdef Vec H1h
    cdef Vec Fp
    cdef Vec Fh
    
    cdef Vec A1
    cdef Vec A2
    
    cdef Vec localH0
    cdef Vec localH1p
    cdef Vec localH1h
    cdef Vec localFp
    cdef Vec localFh
    
    cdef Vec localA1
    cdef Vec localA2
    
    cdef Toolbox toolbox
