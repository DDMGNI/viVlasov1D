'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.Toolbox cimport Toolbox



cdef class PETScMatrix(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t hx2
    cdef np.float64_t hv2
    cdef np.float64_t hx2_inv
    cdef np.float64_t hv2_inv
    
    cdef np.ndarray v
    
    cdef np.float64_t charge
    cdef np.float64_t nu
    
    cdef DA dax
    cdef DA da1
    cdef DA da2
    
    cdef Vec H0
    cdef Vec H1
    cdef Vec H1h
    cdef Vec H2
    cdef Vec H2h
    cdef Vec F
    cdef Vec Fh
    
    cdef Vec P
    cdef Vec N
    cdef Vec U
    cdef Vec E
    cdef Vec A
        
    cdef Vec Ph
    cdef Vec Nh
    cdef Vec Uh
    cdef Vec Eh
    cdef Vec Ah
        
    cdef Vec localH0
    cdef Vec localH1
    cdef Vec localH1h
    cdef Vec localH2
    cdef Vec localH2h
    cdef Vec localF
    cdef Vec localFh
    
    cdef Vec localP
    cdef Vec localN
    cdef Vec localU
    cdef Vec localE
    cdef Vec localA
    
    cdef Vec localPh
    cdef Vec localNh
    cdef Vec localUh
    cdef Vec localEh
    cdef Vec localAh
    
    cdef Toolbox toolbox
