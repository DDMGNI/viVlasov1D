'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport *

from vlasov.solvers.vlasov.PETScVlasovSolver cimport PETScVlasovSolverBase


cdef class PETScVlasovSolver(PETScVlasovSolverBase):

    cdef double a11
    cdef double a12
    cdef double a21
    cdef double a22
    
    cdef np.ndarray f_arr
    cdef np.ndarray h_arr
    
    cdef double[:,:,:] f
    cdef double[:,:,:] h
    
    
    cdef object da2
    
    cdef Vec H11
    cdef Vec H12
    cdef Vec H21
    cdef Vec H22
    
    cdef Vec localK

    cdef Vec localH0
    cdef Vec localH11
    cdef Vec localH12
    cdef Vec localH21
    cdef Vec localH22
    
    
    cpdef update_previous4(self)

