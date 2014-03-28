'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA


cdef class PETScVlasovSolverBase(object):

    cdef double charge
    cdef double nu
    cdef double coll_diff
    cdef double coll_drag
    cdef double regularisation
    
    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec H0
    cdef Vec H1p
    cdef Vec H1h
    cdef Vec H2p
    cdef Vec H2h
    
    cdef Vec Fave
    cdef Vec Fder
    cdef Vec Have
    
    cdef Vec Fp
    cdef Vec Fh
    
    cdef Vec Np
    cdef Vec Up
    cdef Vec Ep
    cdef Vec Ap
    
    cdef Vec Nh
    cdef Vec Uh
    cdef Vec Eh
    cdef Vec Ah
    
    cdef Vec localFave
    cdef Vec localFder
    cdef Vec localHave
    
    cdef Vec localFp
    cdef Vec localFh
    cdef Vec localFd
    
    
    cpdef mult(self, Mat mat, Vec X, Vec Y)
    cpdef snes_mult(self, SNES snes, Vec X, Vec Y)
    cpdef jacobian_mult(self, Vec X, Vec Y)
    
    cpdef function_snes_mult(self, SNES snes, Vec X, Vec Y)
    cpdef function_mult(self, Vec X, Vec Y)
