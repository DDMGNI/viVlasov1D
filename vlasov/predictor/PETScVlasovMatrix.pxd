'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.Toolbox cimport Toolbox


cdef class PETScVlasovMatrix(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t time_fac
    cdef np.float64_t arak_fac
    
    cdef np.ndarray v
    
    cdef np.float64_t nu
    
    cdef DA dax
    cdef DA da1
    
    cdef Vec H0
    
    cdef Vec localB
    cdef Vec localFh
    
    cdef Vec localH0
    cdef Vec localH1
    
    cdef Toolbox toolbox
