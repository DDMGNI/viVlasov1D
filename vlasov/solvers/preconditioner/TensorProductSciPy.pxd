'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

ctypedef np.complex128_t dcomplex

from petsc4py.PETSc cimport Mat, Vec

from pyfftw.pyfftw          cimport FFTW

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from vlasov.solvers.preconditioner.TensorProduct cimport PETScVlasovPreconditioner


cdef class PETScVlasovSolver(PETScVlasovPreconditioner):
    
    cdef list solvers
    
    cdef formSparsePreconditionerMatrix(self, np.complex eigen)
