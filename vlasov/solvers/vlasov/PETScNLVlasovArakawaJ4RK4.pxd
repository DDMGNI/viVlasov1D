'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as npy

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.toolbox.VIDA    cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovSolver cimport PETScVlasovSolverBase


cdef class PETScVlasovSolver(PETScVlasovSolverBase):

    cdef npy.float64_t a11
    cdef npy.float64_t a12
    cdef npy.float64_t a21
    cdef npy.float64_t a22


    cdef VIDA da2
    
    cdef Vec Xd
    
    cdef Vec H11
    cdef Vec H12
    cdef Vec H21
    cdef Vec H22
    
    cdef Vec F1
    cdef Vec F2
    cdef Vec P1
    cdef Vec P2
    
    cdef Vec localXd
    
    cdef Vec localH11
    cdef Vec localH12
    cdef Vec localH21
    cdef Vec localH22

    cdef Vec localF1
    cdef Vec localF2
    cdef Vec localP1
    cdef Vec localP2
    
    cdef npy.ndarray xd
    
    cdef npy.ndarray h11
    cdef npy.ndarray h12
    cdef npy.ndarray h21
    cdef npy.ndarray h22

    cdef npy.ndarray f1
    cdef npy.ndarray f2
    cdef npy.ndarray p1
    cdef npy.ndarray p2


    cpdef update_previous4(self, Vec F1, Vec F2, Vec P1, Vec P2, Vec Pext1, Vec Pext2, Vec N, Vec U, Vec E)

    cdef get_data_arrays(self)

