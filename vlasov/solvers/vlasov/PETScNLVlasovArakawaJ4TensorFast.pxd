'''
Created on Jul 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as npy

from petsc4py.PETSc         cimport Vec

from pyfftw.pyfftw          cimport FFTW

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA

from vlasov.solvers.vlasov.PETScVlasovPreconditioner cimport PETScVlasovPreconditioner


cdef class PETScVlasovSolver(PETScVlasovPreconditioner):
    
    cdef npy.ndarray matrices
    cdef npy.ndarray rhs
    cdef npy.ndarray pivots
    
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
    
    
    cdef call_zgbtrf(self, npy.ndarray A, npy.ndarray IPIV)
    
    cdef call_zgbtrs(self, npy.ndarray[npy.complex128_t, ndim=2] matrix,
                           npy.ndarray[npy.complex128_t, ndim=2] rhs,
                           npy.ndarray[npy.int64_t, ndim=1] pivots)    
    
    cdef formBandedPreconditionerMatrix(self, npy.ndarray matrix, npy.complex eigen)
    

cdef extern void zgbtrf(int* M, int* N, int* KL, int* KU, double complex* A, int* LDA, int* IPIV, int* INFO)
cdef extern void zgbtrs(char* TRANS, int* N, int* KL, int* KU, int* NRHS, double complex* A, int* LDA, int* IPIV, double complex* B, int* LDB, int* INFO)
