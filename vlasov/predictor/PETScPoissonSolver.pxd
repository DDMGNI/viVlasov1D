'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport Mat, Vec

from vlasov.VIDA    cimport VIDA


cdef class PETScPoissonSolver(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t poisson_const
    cdef np.float64_t eps
    
    cdef VIDA da1
    cdef VIDA dax
    
    cdef Vec localX
    cdef Vec localF
    
    
