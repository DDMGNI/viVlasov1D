'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc cimport Mat, Vec

from vlasov.toolbox.VIDA    cimport VIDA


cdef class PETScSmoother(object):

    cdef np.uint64_t nx_coar
    cdef np.uint64_t nv_coar
    
    cdef np.uint64_t nx_fine
    cdef np.uint64_t nv_fine
    
    cdef VIDA da_coar
    cdef VIDA da_fine

    cdef Vec localX_coar
    cdef Vec localX_fine

