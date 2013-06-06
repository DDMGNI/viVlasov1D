'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.vi.PETScSolver cimport PETScSolverBase


cdef class PETScArakawaJ1(PETScSolverBase):
    pass
    
#     cdef function(self, Vec Y)
    
