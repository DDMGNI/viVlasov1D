'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from petsc4py.PETSc cimport Mat, Vec


ctypedef void (*f_time_derivative)(TimeDerivative, Vec, Vec)
ctypedef void (*j_time_derivative)(TimeDerivative, Mat)


cdef class TimeDerivative(object):

    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec localF

    cdef f_time_derivative time_derivative_function
    cdef j_time_derivative time_derivative_jacobian
    
    cdef void call_function(self, Vec F, Vec Y)
    cdef void call_jacobian(self, Mat J)
    
    cdef void point(self, Vec F, Vec Y)
    cdef void midpoint(self, Vec F, Vec Y)
    cdef void simpson(self, Vec F, Vec Y)
    
    cdef void arakawa_J1(self, Vec F, Vec Y)
    cdef void arakawa_J2(self, Vec F, Vec Y)
    cdef void arakawa_J4(self, Vec F, Vec Y)

    cdef void point_jacobian(self, Mat J)
    cdef void midpoint_jacobian(self, Mat J)
    cdef void simpson_jacobian(self, Mat J)

    cdef void arakawa_J1_jacobian(self, Mat J)
    cdef void arakawa_J2_jacobian(self, Mat J)
    cdef void arakawa_J4_jacobian(self, Mat J)
