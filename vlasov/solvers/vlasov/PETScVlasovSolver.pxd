'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as npy

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA


cdef class PETScVlasovSolverBase(object):

    cdef npy.float64_t charge
    cdef npy.float64_t nu
    cdef npy.float64_t coll_diff
    cdef npy.float64_t coll_drag
    cdef npy.float64_t regularisation
    
    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec H0
    cdef Vec H1p
    cdef Vec H1h
    cdef Vec H2p
    cdef Vec H2h
    
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
    
    cdef Vec localH0
    cdef Vec localH1p
    cdef Vec localH1h
    cdef Vec localH2p
    cdef Vec localH2h
    
    cdef Vec localFp
    cdef Vec localFh
    cdef Vec localFd
    
    cdef npy.ndarray h0
    cdef npy.ndarray h1p
    cdef npy.ndarray h1h
    cdef npy.ndarray h2p
    cdef npy.ndarray h2h
    
    cdef npy.ndarray fp
    cdef npy.ndarray fh
    cdef npy.ndarray fd
    
    cdef npy.ndarray np
    cdef npy.ndarray up
    cdef npy.ndarray ep
    cdef npy.ndarray ap
    
    cdef npy.ndarray nh
    cdef npy.ndarray uh
    cdef npy.ndarray eh
    cdef npy.ndarray ah
    
    
    cpdef mult(self, Mat mat, Vec X, Vec Y)
    cpdef snes_mult(self, SNES snes, Vec X, Vec Y)
    cpdef jacobian_mult(self, Vec X, Vec Y)
    
    cpdef function_snes_mult(self, SNES snes, Vec X, Vec Y)
    cpdef function_mult(self, Vec X, Vec Y)
    
    cdef get_data_arrays(self)
    cdef get_data_arrays_jacobian(self)
#     cdef get_data_arrays_function(self)
