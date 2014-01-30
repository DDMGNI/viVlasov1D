'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport Mat, Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovSolver cimport PETScVlasovSolverBase


cdef class PETScVlasovSolver(PETScVlasovSolverBase):

    cdef Vec localH0
    cdef Vec localH1p
    cdef Vec localH1h
    cdef Vec localH2p
    cdef Vec localH2h
