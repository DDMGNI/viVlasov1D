'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from petsc4py.PETSc cimport Mat, Vec


ctypedef void (*f_collision_operator)(CollisionOperator, Vec, Vec, Vec, Vec, Vec, Vec, double)
ctypedef void (*j_collision_operator)(CollisionOperator, Mat, Vec, Vec, Vec, Vec, double)


cdef class CollisionOperator(object):

    cdef double coll_freq
    cdef double coll_diff
    cdef double coll_drag
    
    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec localF
    
    cdef f_collision_operator collision_operator_function
    cdef j_collision_operator collision_operator_jacobian
    
    cdef void call_function(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor)
    cdef void call_jacobian(self, Mat J, Vec N, Vec U, Vec E, Vec A, double factor)
        
    cdef void collT_function(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor)
    cdef void collE_function(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor)

    cdef void collT_jacobian(self, Mat J, Vec N, Vec U, Vec E, Vec A, double factor)
    cdef void collE_jacobian(self, Mat J, Vec N, Vec U, Vec E, Vec A, double factor)
