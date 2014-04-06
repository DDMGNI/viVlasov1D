'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

ctypedef np.complex128_t dcomplex

from petsc4py.PETSc cimport Mat, Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from vlasov.solvers.preconditioner.TensorProduct cimport TensorProductPreconditioner


cdef class TensorProductPreconditionerSciPy(TensorProductPreconditioner):
    
    cdef list solvers
    
    cdef formSparsePreconditionerMatrix(self, np.complex eigen)
