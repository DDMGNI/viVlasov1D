'''
Created on May 27, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from numpy cimport ndarray, float64_t
from petsc4py.PETSc cimport DMDA, Vec


cdef class VIDA(DMDA):
    pass