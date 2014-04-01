'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from petsc4py.PETSc cimport Mat, Vec


cdef class PoissonBracket(object):

    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec localF
    cdef Vec localH
    
    cdef void function(self, Vec F, Vec H, Vec Y, double factor)
    cdef void jacobian(self, Mat J, Vec H, double factor)

    cdef void poisson_bracket_array(self, double[:,:] x, double[:,:] h, double[:,:] y, double factor)
    
    cdef double poisson_bracket_point(self, double[:,:] f, double[:,:] h, int i, int j)
