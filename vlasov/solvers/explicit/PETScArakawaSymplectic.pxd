'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA
from vlasov.solvers.components.PoissonBracket cimport PoissonBracket


cdef class PETScArakawaSymplectic(object):
    
    cdef np.uint64_t niter 
    
    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec H0
    cdef Vec H1
    cdef Vec H2
    
    cdef Vec Y
    
    cdef Vec localH0
    cdef Vec localH1
    cdef Vec localH2
    
    cdef Vec localX
    
    cdef PoissonBracket arakawa
