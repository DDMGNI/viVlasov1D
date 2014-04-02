'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython
import pyfftw

import  numpy as np
cimport numpy as np

from scipy.sparse        import diags, eye
from scipy.sparse.linalg import splu
from scipy.fftpack       import fftshift, ifftshift

from pyfftw.interfaces.scipy_fftpack import fft, ifft

from petsc4py import PETSc


cdef class TensorProductPreconditionerFast(TensorProductPreconditioner):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 VIDA da1  not None,
                 Grid grid not None):
        '''
        Constructor
        '''
        
        super().__init__(da1, grid)
        
        cdef int i, j, xs, xe, ys, ye
        
        # get local x ranges for solver
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # get local x ranges for FFT
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Creating FFTW objects.")
        
        # FFTW arrays 
        fftw_in   = np.empty((ye-ys, self.grid.nx),      dtype=np.float64,    order='c')
        fftw_out  = np.empty((ye-ys, self.grid.nx//2+1), dtype=np.complex128, order='c')
        ifftw_in  = np.empty((ye-ys, self.grid.nx//2+1), dtype=np.complex128, order='c')
        ifftw_out = np.empty((ye-ys, self.grid.nx),      dtype=np.float64,    order='c')
        
        # create pyFFTW plans
        self.fftw_plan  = pyfftw.FFTW(fftw_in,  fftw_out,  axes=(1,), direction='FFTW_FORWARD',  flags=('FFTW_UNALIGNED','FFTW_DESTROY_INPUT'))
        self.ifftw_plan = pyfftw.FFTW(ifftw_in, ifftw_out, axes=(1,), direction='FFTW_BACKWARD', flags=('FFTW_UNALIGNED','FFTW_DESTROY_INPUT'))
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Computing eigenvalues.")
        
        # eigenvalues
        eigen = np.empty(self.grid.nx, dtype=np.complex128)
        
        for i in range(0, self.grid.nx):
            eigen[i] = np.exp(2.j * np.pi * float(i) / self.grid.nx * (self.grid.nx-1)) \
                     - np.exp(2.j * np.pi * float(i) / self.grid.nx)
        
        eigen[:] = ifftshift(eigen)
        
        
        # LAPACK parameters
        self.M = self.grid.nv
        self.N = self.grid.nv
        self.KL = 1
        self.KU = 1
        self.NRHS = 1
        self.LDA  = 4
        self.LDB  = self.grid.nv
        self.T = 'N'
        
        
        # get local x ranges for solver
        (ys, ye), (xs, xe) = self.cay.getRanges()
        
        # matrices, rhs, pivots
        self.matrices = np.zeros((4, self.grid.nv, xe-xs), dtype=np.cdouble, order='F')
        self.pivots   = np.empty((self.grid.nv, xe-xs), dtype=np.int32, order='F')
        
        # build matrices
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Creating Preconditioner Matrices.")
        
        for i in range(0, xe-xs):
            self.formBandedPreconditionerMatrix(self.matrices[:,:,i], eigen[i+xs])
         
        # LU decompositions
        if PETSc.COMM_WORLD.getRank() == 0:
            print("LU Decomposing Preconditioner Matrices.")
         
        for i in range(0, xe-xs):
            if self.call_zgbtrf(self.matrices[:,:,i], self.pivots[:,i]) != 0:
                print("   ERROR in LU Decomposition.")
         
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Preconditioner Initialisation done.")
            print("")
    
    
#     def __dealloc__(self):
#         del self.fftw_plan
#         del self.ifftw_plan
        
    
    cdef fft(self, Vec X, Vec Y):
        # Fourier Transform for each v
        
#         # This code uses some complicated casts to call
#         # the pyFFTW wrapper around FFTW.
#         (xs, xe), (ys, ye) = self.dax.getRanges()
#          
#         dshape = (ye-ys, xe-xs)
#           
#         (xs, xe), (ys, ye) = self.cax.getRanges()
#          
#         cshape = (ye-ys, xe-xs)
#           
#         cdef np.ndarray[double,    ndim=2] x = X.getArray().reshape(dshape, order='c')
#         cdef np.ndarray[np.complex128_t, ndim=2] y = np.empty((ye-ys, self.grid.nx//2+1), dtype=np.complex128, order='c')
#          
#         self.fftw_plan(input_array=x, output_array=y)
#  
#         (<dcomplex[:(ye-ys),:(xe-xs)]> np.PyArray_DATA(Y.getArray()))[...] = y
        
        # This code calls directly into FFTW passing the array
        # buffers of the input and output vectors.
        # The FFTW plan is still setup using pyFFTW.
        # Be careful to allow for unaligned input/output arrays.
        fftw_execute_dft_r2c(<fftw_plan>self.fftw_plan.__plan,
                             <double*>np.PyArray_DATA(X.getArray()),
                             <cdouble*>np.PyArray_DATA(Y.getArray()))

#         cdef void* fftw_planp  = self.fftw_plan.__plan
#         cdef void* fftw_input  = np.PyArray_DATA(X.getArray())
#         cdef void* fftw_output = np.PyArray_DATA(Y.getArray())
#         
#         with nogil:
#             fftw_execute_dft_r2c(<fftw_plan>fftw_planp,
#                                  <double*>fftw_input,
#                                  <double complex*>fftw_output)

        

    
    cdef ifft(self, Vec X, Vec Y):
        # inverse Fourier Transform for each v
        
#         # This code uses some complicated casts to call
#         # the pyFFTW wrapper around FFTW.
#         (xs, xe), (ys, ye) = self.dax.getRanges()
#              
#         dshape = (ye-ys, xe-xs)
#               
#         (xs, xe), (ys, ye) = self.cax.getRanges()
#              
#         cshape = (ye-ys, xe-xs)
#               
#         cdef np.ndarray[double,    ndim=2] y = Y.getArray().reshape(dshape, order='c')
#         cdef np.ndarray[np.complex128_t, ndim=2] x  = np.empty(((ye-ys),(xe-xs)), dtype=np.complex128) 
#         x[...] = (<dcomplex[:(ye-ys),:(xe-xs)]> np.PyArray_DATA(X.getArray()))
#              
#         self.ifftw_plan(input_array=x, output_array=y)
        
        # This code calls directly into FFTW passing the array
        # buffers of the input and output vectors.
        # The FFTW plan is still setup using pyFFTW.
        # Be careful to allow for unaligned input/output arrays.
        fftw_execute_dft_c2r(<fftw_plan>self.ifftw_plan.__plan,
                             <cdouble*>np.PyArray_DATA(X.getArray()),
                             <double*>np.PyArray_DATA(Y.getArray()))
        
        Y.scale(1./float(self.grid.nx))
        
#         cdef void* ifftw_planp  = self.ifftw_plan.__plan
#         cdef void* ifftw_input  = np.PyArray_DATA(X.getArray())
#         cdef void* ifftw_output = np.PyArray_DATA(Y.getArray())
#         
#         with nogil:
#             fftw_execute_dft_c2r(<fftw_plan>ifftw_planp,
#                                  <double complex*>ifftw_input,
#                                  <double*>ifftw_output)


    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef solve(self, Vec X):
        # solve system for each x
        
        cdef int i, j
        cdef int xe, xs, ye, ys
        
        (ys, ye), (xs, xe) = self.cay.getRanges()
        
        assert ys == 0
        assert ye == self.grid.nv
        
        cdef dcomplex[:,:] cx = (<dcomplex[:(xe-xs),:(ye-ys)]> np.PyArray_DATA(X.getArray()))
        
        for i in range(0, xe-xs):
            self.call_zgbtrs(self.matrices[:,:,i], cx[i,:], self.pivots[:,i])
        
    
    cdef call_zgbtrf(self, dcomplex[:,:] matrix, int[:] pivots):
        cdef int INFO = 0
        
        zgbtrf(&self.M, &self.N, &self.KL, &self.KU, &matrix[0,0], &self.LDA, &pivots[0], &INFO)
        
        return INFO
     
 
    cdef call_zgbtrs(self, dcomplex[:,:] matrix, dcomplex[:] rhs, int[:] pivots):
        
        cdef int INFO = 0
         
        zgbtrs(&self.T, &self.N, &self.KL, &self.KU, &self.NRHS, &matrix[0,0], &self.LDA, &pivots[0], &rhs[0], &self.LDB, &INFO)
        
        return INFO
        
    
    cdef formBandedPreconditionerMatrix(self, dcomplex[:,:] matrix, dcomplex eigen):
        cdef int j
        
        cdef double[:] v = self.grid.v
        
        cdef double arak_fac_J1 = 0.5 / (12. * self.grid.hx * self.grid.hv)
        
        
        cdef dcomplex[:] diagm = np.zeros(self.grid.nv, dtype=np.cdouble)
        cdef dcomplex[:] diag  = np.ones (self.grid.nv, dtype=np.cdouble)
        cdef dcomplex[:] diagp = np.zeros(self.grid.nv, dtype=np.cdouble)
        
        for j in range(2, self.grid.nv-2):
            diagm[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] - self.grid.hv2 ) * arak_fac_J1
            diag [j] = eigen * 4.0 * self.grid.hv * v[j] * arak_fac_J1 + self.grid.ht_inv
            diagp[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] + self.grid.hv2 ) * arak_fac_J1
        
        matrix[1, 1:  ] = diagp[:-1]
        matrix[2,  :  ] = diag[:]
        matrix[3,  :-1] = diagm[1:]
        
