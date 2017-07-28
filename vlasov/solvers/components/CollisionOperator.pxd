'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid cimport Grid
from vlasov.toolbox.VIDA cimport *

from petsc4py.PETSc cimport Mat, Vec


cdef class CollisionOperator(object):

    cdef double coll_freq
    cdef double coll_diff
    cdef double coll_drag
    
    cdef object da1
    cdef Grid grid
    
    cdef Vec localF
    
    cdef void function(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor)
    cdef void jacobian(self, Mat J, Vec N, Vec U, Vec E, Vec A, double factor)
