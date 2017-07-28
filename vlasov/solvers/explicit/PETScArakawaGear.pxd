'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport *

from vlasov.solvers.explicit.PETScExplicitSolver cimport PETScExplicitSolver


cdef class PETScArakawaGear(PETScExplicitSolver):
    
    cdef Vec H1h1
    cdef Vec H1h2
    cdef Vec H1h3
    cdef Vec H1h4
    
    cdef Vec H2h1
    cdef Vec H2h2
    cdef Vec H2h3
    cdef Vec H2h4
    
    cdef Vec Fh1
    cdef Vec Fh2
    cdef Vec Fh3
    cdef Vec Fh4
    
    cdef Vec localH1h1
    cdef Vec localH1h2
    cdef Vec localH1h3
    cdef Vec localH1h4

    cdef Vec localH2h1
    cdef Vec localH2h2
    cdef Vec localH2h3
    cdef Vec localH2h4

    cdef Vec localFh1
    cdef Vec localFh2
    cdef Vec localFh3
    cdef Vec localFh4
