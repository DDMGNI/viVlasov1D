'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport IS, Scatter, Vec

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovSolver cimport PETScVlasovSolverBase


cdef class PETScVlasovPreconditioner(PETScVlasovSolverBase):
    
    cdef VIDA dax
    cdef VIDA day
    
    cdef VIDA cax
    cdef VIDA cay
    
    cdef IS cxindices
    cdef IS cyindices
    
    cdef Scatter xyScatter
    cdef Scatter yxScatter
    
    cdef Vec B
    cdef Vec X
    cdef Vec F
    cdef Vec Z
    
    cdef Vec Ffft
    cdef Vec Bfft
    cdef Vec Zfft
    
    
    cpdef jacobian(self, Vec F, Vec Y)
    cpdef function(self, Vec F, Vec Y)

    cdef tensorProduct(self, Vec X, Vec Y)
    
    cdef copy_da1_to_dax(self, Vec X, Vec Y)
    cdef copy_dax_to_da1(self, Vec X, Vec Y)
    
    cdef copy_cax_to_cay(self, Vec X, Vec Y)
    cdef copy_cay_to_cax(self, Vec X, Vec Y)
    
    cdef fft (self, Vec X, Vec Y)
    cdef ifft(self, Vec X, Vec Y)

    cdef solve(self, Vec X)
    
    cdef jacobianSolver(self, Vec F, Vec Y)
    cdef functionSolver(self, Vec F, Vec Y)
