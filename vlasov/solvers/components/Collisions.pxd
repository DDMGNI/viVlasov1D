'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from petsc4py.PETSc cimport Vec


cdef class Collisions(object):

    cdef double coll_freq
    cdef double coll_diff
    cdef double coll_drag
    
    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec localF
    
    
    cpdef collT(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor)
    cpdef collE(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor)
