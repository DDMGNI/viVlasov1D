'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport Mat, Vec

from vlasov.VIDA    cimport VIDA
from vlasov.Toolbox cimport Toolbox


cdef class PETScFunction(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t hx2
    cdef np.float64_t hx2_inv
    
    cdef np.float64_t hv2
    cdef np.float64_t hv2_inv
    
    cdef np.float64_t charge
    cdef np.float64_t nu
    
    cdef np.ndarray v
    
    cdef VIDA dax
    cdef VIDA da1
    cdef VIDA da2
    
    cdef Vec H0
    cdef Vec H1p
    cdef Vec H1h
    cdef Vec H2p
    cdef Vec H2h
    cdef Vec Fp
    cdef Vec Fh
    
    cdef Vec Pp
    cdef Vec Np
    cdef Vec NUp
    cdef Vec NEp
    cdef Vec Up
    cdef Vec Ep
    cdef Vec Ap
    
    cdef Vec Ph
    cdef Vec Nh
    cdef Vec NUh
    cdef Vec NEh
    cdef Vec Uh
    cdef Vec Eh
    cdef Vec Ah
    
    cdef Vec Nc
    cdef Vec NUc
    cdef Vec NEc
    
    cdef Vec localH0
    cdef Vec localH1p
    cdef Vec localH1h
    cdef Vec localH2p
    cdef Vec localH2h
    cdef Vec localFp
    cdef Vec localFh
    
    cdef Vec localPp
    cdef Vec localNp
    cdef Vec localNUp
    cdef Vec localNEp
    cdef Vec localUp
    cdef Vec localEp
    cdef Vec localAp
    
    cdef Vec localPh
    cdef Vec localNh
    cdef Vec localNUh
    cdef Vec localNEh
    cdef Vec localUh
    cdef Vec localEh
    cdef Vec localAh
    
    cdef Vec localNc
    cdef Vec localNUc
    cdef Vec localNEc
    
    
    cdef Toolbox toolbox

    
    cdef matrix_mult(self, Vec Y)
    
