'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as npy

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovSolver cimport PETScVlasovSolverBase


cdef class PETScVlasovSolver(PETScVlasovSolverBase):

    cdef Vec H11
    cdef Vec H21
    
    cdef Vec localK
    
    cdef Vec localH11
    cdef Vec localH21
    
    
    cpdef update_previous2(self)

