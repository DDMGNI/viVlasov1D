'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from petsc4py.PETSc cimport Mat, Vec


ctypedef void (*f_regularisation)(Regularisation, Vec, Vec, double)
ctypedef void (*j_regularisation)(Regularisation, Mat, double)


cdef class Regularisation(object):

    cdef double epsilon
    
    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec localF
    
    cdef f_regularisation call_regularisation_function
    cdef j_regularisation call_regularisation_jacobian
    
    cdef void call_function(self, Vec F, Vec Y, double factor)
    cdef void call_jacobian(self, Mat J, double factor)
    
    cdef void regularisation_function(self, Vec F, Vec Y, double factor)
    cdef void regularisation_jacobian(self, Mat J, double factor)
