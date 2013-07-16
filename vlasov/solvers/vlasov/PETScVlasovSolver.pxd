'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as npy

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.toolbox.VIDA    cimport VIDA
from vlasov.toolbox.Toolbox cimport Toolbox


cdef class PETScVlasovSolverBase(object):

    cdef npy.uint64_t  nx
    cdef npy.uint64_t  nv
    
    cdef npy.float64_t ht
    cdef npy.float64_t hx
    cdef npy.float64_t hv
    
    cdef npy.float64_t ht_inv
    cdef npy.float64_t hx_inv
    cdef npy.float64_t hv_inv
    
    cdef npy.float64_t hx2
    cdef npy.float64_t hx2_inv
    
    cdef npy.float64_t hv2
    cdef npy.float64_t hv2_inv
    
    cdef npy.float64_t charge
    cdef npy.float64_t nu
    cdef npy.float64_t coll_diff
    cdef npy.float64_t coll_drag
    cdef npy.float64_t regularisation
    
    cdef npy.ndarray v
    
    
    cdef VIDA dax
    cdef VIDA da1
    
    cdef Vec Fp
    cdef Vec Fh
    cdef Vec Fd
    
    cdef Vec H0
    cdef Vec H1p
    cdef Vec H1h
    cdef Vec H2p
    cdef Vec H2h
    
    cdef Vec Pp
    cdef Vec Np
    cdef Vec Up
    cdef Vec Ep
    cdef Vec Ap
    
    cdef Vec Ph
    cdef Vec Nh
    cdef Vec Uh
    cdef Vec Eh
    cdef Vec Ah
    
    cdef Vec localFp
    cdef Vec localFh
    cdef Vec localFd
    
    cdef Vec localH0
    cdef Vec localH1p
    cdef Vec localH1h
    cdef Vec localH1d
    cdef Vec localH2p
    cdef Vec localH2h
    
    cdef Vec localPp
    cdef Vec localNp
    cdef Vec localUp
    cdef Vec localEp
    cdef Vec localAp
    
    cdef Vec localPh
    cdef Vec localNh
    cdef Vec localUh
    cdef Vec localEh
    cdef Vec localAh
    
    cdef npy.ndarray fp
    cdef npy.ndarray fh
    cdef npy.ndarray fd
    
    cdef npy.ndarray h0
    cdef npy.ndarray h1p
    cdef npy.ndarray h1h
    cdef npy.ndarray h2p
    cdef npy.ndarray h2h
    
    cdef npy.ndarray pp
    cdef npy.ndarray np
    cdef npy.ndarray up
    cdef npy.ndarray ep
    cdef npy.ndarray ap
    
    cdef npy.ndarray ph
    cdef npy.ndarray nh
    cdef npy.ndarray uh
    cdef npy.ndarray eh
    cdef npy.ndarray ah
    
    cdef Toolbox toolbox
    
    
    cpdef update_history (self, Vec F, Vec P, Vec Pext, Vec N, Vec U, Vec E)
    cpdef update_previous(self, Vec F, Vec P, Vec Pext, Vec N, Vec U, Vec E)
    cpdef update_delta   (self, Vec F)
    
    cpdef mult(self, Mat mat, Vec X, Vec Y)
    cpdef snes_mult(self, SNES snes, Vec X, Vec Y)
    cpdef jacobian_mult(self, Vec X, Vec Y)
    
    cpdef function_snes_mult(self, SNES snes, Vec X, Vec Y)
    cpdef function_mult(self, Vec X, Vec Y)
    
    cdef get_data_arrays(self)
    cdef get_data_arrays_jacobian(self)
    cdef get_data_arrays_function(self)
