'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid cimport Grid
from vlasov.toolbox.VIDA cimport *

from petsc4py.PETSc cimport Mat, Vec


cdef class Regularisation(object):

    cdef double epsilon
    
    cdef object da1
    cdef Grid grid
    
    cdef Vec localF
    
    cdef void function(self, Vec F, Vec Y, double factor)
    cdef void jacobian(self, Mat J, double factor)

    cdef void regularisation_function(self, Vec F, Vec Y, double factor)
    cdef void regularisation_jacobian(self, Mat J, double factor)
