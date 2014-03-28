'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from petsc4py.PETSc cimport Vec


cdef class Regularisation(object):

    cdef double epsilon
    
    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec localF
    
    
    cpdef regularisation(self, Vec F, Vec Y, double factor)
