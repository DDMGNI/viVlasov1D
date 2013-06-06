'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.vi.PETScSolver cimport PETScSolverBase


cdef class PETScArakawaJ2(PETScSolverBase):
    pass
    
#     cdef function(self, Vec Y)
    
