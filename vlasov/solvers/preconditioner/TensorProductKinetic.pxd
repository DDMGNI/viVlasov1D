'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport IS, Scatter, Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport *

from vlasov.solvers.preconditioner.TensorProduct cimport *


cdef class TensorProductPreconditionerKinetic(TensorProductPreconditioner):
    
    cdef fft (self, Vec X, Vec Y)
    cdef ifft(self, Vec X, Vec Y)
