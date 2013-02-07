'''
Created on Jul 10, 2012

@author: mkraus
'''

cimport cython

cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.vi.Toolbox cimport Toolbox


cdef class PETScJacobianMatrixFree(object):
    
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
    cdef Vec H2
    cdef Vec H2h
    cdef Vec Hp
    cdef Vec Hh
    cdef Vec Fp
    cdef Vec Fh
    
    cdef Vec A1d
    cdef Vec A2d
    cdef Vec A3d
    cdef Vec A1p
    cdef Vec A2p
    cdef Vec A3p
    
    cdef Vec Nd
    cdef Vec Ud
    cdef Vec Ed
    cdef Vec Np
    cdef Vec Up
    cdef Vec Ep
    
    cdef Vec localH0
    cdef Vec localH2
    cdef Vec localH2h
    cdef Vec localHd
    cdef Vec localHp
    cdef Vec localHh
    cdef Vec localFd
    cdef Vec localFp
    cdef Vec localFh
    cdef Vec localPd
    
    cdef Vec localA1d
    cdef Vec localA2d
    cdef Vec localA3d
    cdef Vec localA1p
    cdef Vec localA2p
    cdef Vec localA3p
    
    cdef Vec localNd
    cdef Vec localUd
    cdef Vec localEd
    cdef Vec localNp
    cdef Vec localUp
    cdef Vec localEp
    
    cdef Toolbox toolbox


    cdef np.float64_t collN1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] nd,
                                   np.ndarray[np.float64_t, ndim=1] np,
                                   np.ndarray[np.float64_t, ndim=1] ud,
                                   np.ndarray[np.float64_t, ndim=1] up,
                                   np.uint64_t i, np.uint64_t j)
    
    cdef np.float64_t collN2(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] nd,
                                   np.ndarray[np.float64_t, ndim=1] np,
                                   np.ndarray[np.float64_t, ndim=1] ud,
                                   np.ndarray[np.float64_t, ndim=1] up,
                                   np.ndarray[np.float64_t, ndim=1] ed,
                                   np.ndarray[np.float64_t, ndim=1] ep,
                                   np.uint64_t i, np.uint64_t j)
    
