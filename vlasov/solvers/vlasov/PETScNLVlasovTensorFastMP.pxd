'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc      cimport Mat, Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

cimport vlasov.solvers.preconditioner.TensorProductFast


cdef class PETScVlasovSolver(vlasov.solvers.preconditioner.TensorProductFast.PETScVlasovSolver):
    pass
