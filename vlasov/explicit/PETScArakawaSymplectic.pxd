'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.toolbox.VIDA    cimport VIDA
from vlasov.toolbox.Arakawa cimport Arakawa


cdef class PETScArakawaSymplectic(object):
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef VIDA da1
    
    cdef np.ndarray v
    
    cdef Vec H0
    
    cdef Vec Y
    
    cdef Vec localX
    cdef Vec localH
    
    cdef Arakawa arakawa
