'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport Mat, Vec

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovSolver cimport PETScVlasovSolverBase
# cimport vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4


cdef class PETScVlasovSolver(PETScVlasovSolverBase):
    
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
    
    cdef dict pmats
    cdef dict solvers
    
#     cdef vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4.PETScVlasovSolver solver
    

    cdef copy_da1_to_dax(self, Vec X, Vec Y)
    cdef copy_dax_to_da1(self, Vec X, Vec Y)
    cdef copy_dax_to_day(self, Vec X, Vec Y)
    cdef copy_day_to_dax(self, Vec X, Vec Y)
    
    cdef fft (self, Vec X, Vec YR, Vec YI)
    cdef ifft(self, Vec XR, Vec XI, Vec Y)

    cdef solve(self, Vec XR, Vec XI, Vec YR, Vec YI)
    
    cdef formPreconditionerMatrix(self)
    
    cdef jacobianArakawaJ1(self, Vec F, Vec Y)
    cdef functionArakawaJ1(self, Vec F, Vec Y)
    
    cdef jacobianArakawaJ4(self, Vec F, Vec Y)
    cdef functionArakawaJ4(self, Vec F, Vec Y)
