'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as np

from libc.stdint cimport intptr_t

ctypedef np.complex128_t dcomplex


from petsc4py.PETSc         cimport Vec

from pyfftw.pyfftw          cimport FFTW

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovPreconditioner cimport PETScVlasovPreconditioner


cdef class PETScVlasovSolver(PETScVlasovPreconditioner):
    
    cdef dcomplex[:,:,:] matrices
    cdef int[:,:] pivots
    
    cdef FFTW fftw_plan
    cdef FFTW ifftw_plan
    
    
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
    

cdef extern void zgbtrf(int* M, int* N, int* KL, int* KU, double complex* A, int* LDA, int* IPIV, int* INFO)
cdef extern void zgbtrs(char* TRANS, int* N, int* KL, int* KU, int* NRHS, double complex* A, int* LDA, int* IPIV, double complex* B, int* LDB, int* INFO)


cdef extern from 'pyfftw_complex.h':
    ctypedef double cdouble[2]

cdef extern from 'fftw3.h':
    ctypedef struct fftw_plan_struct:
        pass

    ctypedef fftw_plan_struct* fftw_plan

    void fftw_execute_dft_r2c(fftw_plan, double* _in, cdouble* _out) nogil
    void fftw_execute_dft_c2r(fftw_plan, cdouble* _in, double* _out) nogil    

