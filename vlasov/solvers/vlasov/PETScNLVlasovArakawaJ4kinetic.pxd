'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport Mat, Vec

cimport vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4


cdef class PETScVlasovSolverKinetic(vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4.PETScVlasovSolver):
    pass
