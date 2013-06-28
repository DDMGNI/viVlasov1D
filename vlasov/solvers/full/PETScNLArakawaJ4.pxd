'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport Mat, Vec

from vlasov.solvers.full.PETScFullSolver cimport PETScFullSolverBase


cdef class PETScSolver(PETScFullSolverBase):
    pass

#     cdef function(self, Vec Y)
#     cdef jacobian(self, Vec Y)
    
