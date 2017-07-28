'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport IS, Scatter, Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport *

# ctypedef np.complex128_t dcomplex
# ctypedef npy_complex128 dcomplex
ctypedef double complex dcomplex
ctypedef double cdouble[2]


cdef class TensorProductPreconditioner(object):
    
    cdef object da1
    cdef Grid   grid
    
    cdef object dax
    cdef object day
    
    cdef object cax
    cdef object cay
    
    cdef IS d1Indices
    cdef IS dxIndices
    cdef IS dyIndices
    cdef IS cxIndices
    cdef IS cyIndices
    
    cdef Scatter d1xScatter
    cdef Scatter dx1Scatter
    cdef Scatter d1yScatter
    cdef Scatter dy1Scatter
    cdef Scatter cxyScatter
    cdef Scatter cyxScatter
    
    cdef dcomplex[:]     eigen
    cdef dcomplex[:,:,:] matrices
    
    cdef Vec B
    cdef Vec X
    cdef Vec F
    cdef Vec Z
    
    cdef Vec Ffft
    cdef Vec Bfft
    cdef Vec Cfft
    cdef Vec Zfft
    
    cdef copy_da1_to_dax(self, Vec X, Vec Y)
    cdef copy_dax_to_da1(self, Vec X, Vec Y)
    
    cdef copy_da1_to_day(self, Vec X, Vec Y)
    cdef copy_day_to_da1(self, Vec X, Vec Y)
    
    cdef copy_cax_to_cay(self, Vec X, Vec Y)
    cdef copy_cay_to_cax(self, Vec X, Vec Y)
    
    cdef update_matrices(self)
    cdef tensorProduct(self, Vec X, Vec Y)
    cdef solve(self, Vec X)
