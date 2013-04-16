'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

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
    
    cdef DA dax
    cdef DA da1
    cdef DA da2
    
    cdef Vec H0
    cdef Vec H1p
    cdef Vec H1h
    cdef Vec H2p
    cdef Vec H2h
    cdef Vec Fp
    cdef Vec Fh
    
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
    
    cdef Vec localH0
    cdef Vec localH1p
    cdef Vec localH1h
    cdef Vec localH2p
    cdef Vec localH2h
    cdef Vec localFp
    cdef Vec localFh
    
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
    
    
    cdef Toolbox toolbox

