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

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA


cdef class PETScPoissonSolver(object):

    cdef double charge
    
    cdef VIDA dax
    cdef Grid grid
    
    cdef Vec localX
    cdef Vec localN
    
    
