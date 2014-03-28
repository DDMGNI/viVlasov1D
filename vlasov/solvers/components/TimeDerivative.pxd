'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from petsc4py.PETSc cimport Vec


cdef class TimeDerivative(object):

    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec localF


    cdef void arakawa_J1(self, Vec F, Vec Y)
    cdef void arakawa_J2(self, Vec F, Vec Y)
    cdef void arakawa_J4(self, Vec F, Vec Y)
    cdef void midpoint(self, Vec F, Vec Y)
    cdef void simpson(self, Vec F, Vec Y)
    cdef void time_derivative(self, Vec F, Vec Y)
