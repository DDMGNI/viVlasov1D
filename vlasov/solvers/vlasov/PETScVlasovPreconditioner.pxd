'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport Vec

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovSolver cimport PETScVlasovSolverBase


cdef class PETScVlasovPreconditioner(PETScVlasovSolverBase):
    
    cdef VIDA dax
    cdef VIDA day
    
    cdef Vec B
    cdef Vec X
    cdef Vec F
    cdef Vec FfftR
    cdef Vec FfftI
    cdef Vec BfftR
    cdef Vec BfftI
    cdef Vec CfftR
    cdef Vec CfftI
    cdef Vec ZfftR
    cdef Vec ZfftI
    cdef Vec Z
    
    cpdef jacobian(self, Vec F, Vec Y)
    cpdef function(self, Vec F, Vec Y)

    cdef tensorProduct(self, Vec X, Vec Y)
    
    cdef copy_da1_to_dax(self, Vec X, Vec Y)
    cdef copy_dax_to_da1(self, Vec X, Vec Y)
    cdef copy_dax_to_day(self, Vec X, Vec Y)
    cdef copy_day_to_dax(self, Vec X, Vec Y)
    
    cdef fft (self, Vec X, Vec YR, Vec YI)
    cdef ifft(self, Vec XR, Vec XI, Vec Y)

    cdef solve(self, Vec XR, Vec XI, Vec YR, Vec YI)
    
    cdef jacobianSolver(self, Vec F, Vec Y)
    cdef functionSolver(self, Vec F, Vec Y)
