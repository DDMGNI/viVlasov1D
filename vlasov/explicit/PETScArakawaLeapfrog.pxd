'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.toolbox.VIDA    cimport VIDA
from vlasov.toolbox.Arakawa cimport Arakawa


cdef class PETScArakawaLeapfrog(object):
    
    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef VIDA da1
    
    cdef np.ndarray v
    
    cdef Vec H0
    
    cdef Vec H1h1
    cdef Vec H1h2
    
    cdef Vec Fh1
    cdef Vec Fh2
    
    cdef Vec X1
    cdef Vec X2
    cdef Vec X3
    cdef Vec X4
    
    cdef Vec localX
    cdef Vec localX1
    cdef Vec localX2
    cdef Vec localX3
    cdef Vec localX4
    
    cdef Vec localH0
    
    cdef Vec localH1h1
    cdef Vec localH1h2

    cdef Vec localFh1
    cdef Vec localFh2
    
    cdef Arakawa arakawa
