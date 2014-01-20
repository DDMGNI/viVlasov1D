'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA
from vlasov.toolbox.Arakawa cimport Arakawa


cdef class PETScExplicitSolver(object):
    
    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec H0
    cdef Vec H1
    cdef Vec H2
    
    cdef Vec X1
    cdef Vec X2
    cdef Vec X3
    cdef Vec X4
    
    cdef Vec localX
    cdef Vec localX1
    cdef Vec localX2
    cdef Vec localX3
    cdef Vec localX4
    
    cdef Vec localH0
    cdef Vec localH1
    cdef Vec localH2
    
    cdef Arakawa arakawa
