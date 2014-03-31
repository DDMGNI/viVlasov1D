'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport Mat, Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovSolver cimport PETScVlasovSolverBase


cdef class PETScVlasovSolver(PETScVlasovSolverBase):

    cdef VIDA da2
    
    cdef Vec Ft
    cdef Vec Gt
    cdef Vec Gp
    cdef Vec Gh
    cdef Vec Gave
    
    cdef Vec localK
    cdef Vec localGp
    cdef Vec localGh
    cdef Vec localGd
    cdef Vec localGave
