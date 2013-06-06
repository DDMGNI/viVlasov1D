'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport Mat, Vec

from vlasov.vi.PETScSolver cimport PETScSolverBase


cdef class PETScSolver(PETScSolverBase):
    pass

#     cdef function(self, Vec Y)
#     cdef jacobian(self, Vec Y)
    
