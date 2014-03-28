'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from petsc4py.PETSc cimport Vec


cdef class PoissonBracket(object):

    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec localF
    cdef Vec localH
    
    
    cpdef arakawa_J1(self, Vec X, Vec H, Vec Y, double factor)
    cpdef arakawa_J2(self, Vec X, Vec H, Vec Y, double factor)
    cpdef arakawa_J4(self, Vec X, Vec H, Vec Y, double factor)
    
    cpdef arakawa_J1_array(self, double[:,:] x, double[:,:] y, double[:,:] h, double factor)
    cpdef arakawa_J2_array(self, double[:,:] x, double[:,:] y, double[:,:] h, double factor)
    cpdef arakawa_J4_array(self, double[:,:] x, double[:,:] y, double[:,:] h, double factor)
    
    cdef double arakawa_J1_point(self, double[:,:] f, double[:,:] h, int i, int j)
    cdef double arakawa_J2_point(self, double[:,:] f, double[:,:] h, int i, int j)
    cdef double arakawa_J4_point(self, double[:,:] f, double[:,:] h, int i, int j)
    
