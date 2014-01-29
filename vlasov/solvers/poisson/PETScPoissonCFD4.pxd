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

from vlasov.toolbox.VIDA    cimport VIDA


cdef class PETScPoissonSolver(object):

    cdef np.uint64_t  nx
    cdef np.float64_t hx
    
    cdef np.float64_t hx2
    cdef np.float64_t hx2_inv
    
    cdef np.float64_t charge
    
    cdef VIDA dax
    
    cdef Vec localX
    cdef Vec localN
    
    
