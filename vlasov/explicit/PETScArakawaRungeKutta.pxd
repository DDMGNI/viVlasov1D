'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA
from vlasov.toolbox.Arakawa cimport Arakawa

from vlasov.explicit.PETScExplicitSolver cimport PETScExplicitSolver


cdef class PETScArakawaRungeKutta(PETScExplicitSolver):
    pass

