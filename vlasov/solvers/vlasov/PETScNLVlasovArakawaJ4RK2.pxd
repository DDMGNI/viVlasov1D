'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as npy

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.toolbox.VIDA    cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovSolver cimport PETScVlasovSolverBase


cdef class PETScVlasovSolver(PETScVlasovSolverBase):

    cdef Vec H11
    cdef Vec H21
    
    cdef Vec F1
    cdef Vec P1
    
    cdef Vec localK
    
    cdef Vec localH11
    cdef Vec localH21
    
    cdef Vec localF1
    cdef Vec localP1
    
    cdef npy.ndarray h11
    cdef npy.ndarray h21
    
    cdef npy.ndarray f1
    cdef npy.ndarray p1


    cpdef update_previous2(self, Vec F1, Vec P1, Vec Pext1, Vec N, Vec U, Vec E)

    cdef get_data_arrays(self)

