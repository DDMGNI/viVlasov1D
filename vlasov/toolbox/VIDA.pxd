'''
Created on May 27, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from cpython cimport bool
from numpy cimport ndarray, float64_t
from petsc4py.PETSc cimport Vec


cdef reshape(object dmda, ndarray vec, bool local)

cpdef getGlobalArray(object dmda, Vec tvec)
cpdef getGlobalArrayRO(object dmda, Vec tvec)
cpdef getLocalArray(object dmda, Vec gvec, Vec lvec)
cpdef getLocalArrayRO(object dmda, Vec gvec, Vec lvec)
