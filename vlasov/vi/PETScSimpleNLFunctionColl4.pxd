'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.vi.Toolbox cimport Toolbox


cdef class PETScFunction(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t hx2
    cdef np.float64_t hx2_inv
    
    cdef np.float64_t hv2
    cdef np.float64_t hv2_inv
    
    cdef np.float64_t charge
    cdef np.float64_t nu
    
    cdef np.ndarray v
    
    cdef DA dax
    cdef DA da1
    cdef DA da2
    
    cdef Vec H0
    cdef Vec H2
    cdef Vec H2h
    cdef Vec Fh
    cdef Vec Hh
    cdef Vec Ph
    
    cdef Vec localH0
    cdef Vec localF
    cdef Vec localFh
    cdef Vec localH
    cdef Vec localHh
    cdef Vec localH2
    cdef Vec localH2h
    cdef Vec localP
    cdef Vec localPh
    
    cdef Toolbox toolbox
