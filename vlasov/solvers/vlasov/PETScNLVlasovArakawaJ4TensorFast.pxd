'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as npy

from libc.stdint cimport intptr_t

ctypedef npy.complex128_t dcomplex
ctypedef double cdouble[2]


from petsc4py.PETSc         cimport Vec

from pyfftw.pyfftw          cimport FFTW

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovPreconditioner cimport PETScVlasovPreconditioner


cdef class PETScVlasovSolver(PETScVlasovPreconditioner):
    
    cdef dcomplex[:,:,:] matrices
    cdef dcomplex[:,:,:] rhs
    cdef int[:,:] pivots
    cdef npy.ndarray rhs_arr
    
    
#     cdef dcomplex[:,:] fftw_in
#     cdef dcomplex[:,:] fftw_out
#     cdef dcomplex[:,:] ifftw_in
#     cdef dcomplex[:,:] ifftw_out
#     cdef cdouble[:,:] fftw_in
#     cdef cdouble[:,:] fftw_out
#     cdef cdouble[:,:] ifftw_in
#     cdef cdouble[:,:] ifftw_out
    cdef npy.ndarray fftw_in
    cdef npy.ndarray fftw_out
    cdef npy.ndarray ifftw_in
    cdef npy.ndarray ifftw_out
    
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
    cdef call_zgbtrs(self, dcomplex[:,:] matrix, dcomplex[:,:] rhs, int[:] pivots)    
    
    cdef formBandedPreconditionerMatrix(self, dcomplex[:,:] matrix, npy.complex eigen)
    

cdef extern void zgbtrf(int* M, int* N, int* KL, int* KU, double complex* A, int* LDA, int* IPIV, int* INFO)
cdef extern void zgbtrs(char* TRANS, int* N, int* KL, int* KU, int* NRHS, double complex* A, int* LDA, int* IPIV, double complex* B, int* LDB, int* INFO)
# cdef extern void zgbtrf(int* M, int* N, int* KL, int* KU, cdouble* A, int* LDA, int* IPIV, int* INFO)
# cdef extern void zgbtrs(char* TRANS, int* N, int* KL, int* KU, int* NRHS, cdouble* A, int* LDA, int* IPIV, cdouble* B, int* LDB, int* INFO)
