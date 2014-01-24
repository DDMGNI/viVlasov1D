'''
Created on May 27, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from cpython cimport bool
from numpy cimport ndarray, float64_t
from petsc4py.PETSc cimport DMDA, Vec


cdef class VIDA(DMDA):

    cdef reshape(self, ndarray vec, bool local)
    
    cpdef getGlobalArray(self, Vec tvec)
    cpdef getLocalArray(self, Vec gvec, Vec lvec)
