'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as npy

from petsc4py.PETSc         cimport Vec

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA

cimport vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4TensorFast


cdef class PETScVlasovSolver(vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4TensorFast.PETScVlasovSolver):
    pass
