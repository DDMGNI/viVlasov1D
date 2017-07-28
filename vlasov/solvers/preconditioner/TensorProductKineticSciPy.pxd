'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

ctypedef np.complex128_t dcomplex

from petsc4py.PETSc cimport Mat, Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport *

from vlasov.solvers.preconditioner.TensorProduct        cimport *
from vlasov.solvers.preconditioner.TensorProductKinetic cimport TensorProductPreconditionerKinetic


cdef class TensorProductPreconditionerKineticSciPy(TensorProductPreconditionerKinetic):
    
#     cdef list solvers
    
    cdef formBandedPreconditionerMatrix(self, dcomplex[:,:] matrix, dcomplex eigen)
#     cdef formSparsePreconditionerMatrix(self, np.complex eigen)
