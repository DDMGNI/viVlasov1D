'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from petsc4py.PETSc         cimport Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport *

from vlasov.solvers.preconditioner.TensorProduct          cimport *
from vlasov.solvers.preconditioner.TensorProductPotential cimport TensorProductPreconditionerPotential


cdef class TensorProductPreconditionerPotentialFast(TensorProductPreconditionerPotential):
    
    cdef object fftw_plan
    cdef object ifftw_plan
    
    cdef int[:,:] pivots
    
    cdef int M
    cdef int N
    cdef int KL
    cdef int KU
    cdef int NRHS
    cdef int LDA
    cdef int LDB
    cdef char T
    
    
    cdef call_zgbtrf(self, dcomplex[:,:] matrix, int[:] pivots)
    cdef call_zgbtrs(self, dcomplex[:,:] matrix, dcomplex[:] rhs, int[:] pivots)    
    
    cdef formBandedPreconditionerMatrix(self, dcomplex[:,:] matrix, dcomplex eigen)
    

cdef extern void zgbtrf(int* M, int* N, int* KL, int* KU, dcomplex* A, int* LDA, int* IPIV, int* INFO)
cdef extern void zgbtrs(char* TRANS, int* N, int* KL, int* KU, int* NRHS, dcomplex* A, int* LDA, int* IPIV, dcomplex* B, int* LDB, int* INFO)


# cdef extern from 'pyfftw_complex.h':
#     ctypedef double cdouble[2]

cdef extern from 'fftw3.h':
    ctypedef struct fftw_plan_struct:
        pass

    ctypedef fftw_plan_struct* fftw_plan

    void fftw_execute_dft_r2c(fftw_plan, double* _in, cdouble* _out) nogil
    void fftw_execute_dft_c2r(fftw_plan, cdouble* _in, double* _out) nogil    

