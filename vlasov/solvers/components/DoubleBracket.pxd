'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from petsc4py.PETSc cimport Mat, Vec

from vlasov.solvers.components.PoissonBracket cimport PoissonBracket


cdef class DoubleBracket(object):

    cdef double coll_freq
    
    cdef PoissonBracket    poisson_bracket
    
    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec bracket
    
    cdef void jacobian(self, Vec F, Vec Fave, Vec Have, Vec Y, double factor)
    cdef void function(self,        Vec Fave, Vec Have, Vec Y, double factor)
