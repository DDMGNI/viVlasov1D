'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.solvers.full.PETScFullSolver cimport PETScFullSolverBase


cdef class PETScArakawaJ2(PETScFullSolverBase):
    pass
    
#     cdef function(self, Vec Y)
    
