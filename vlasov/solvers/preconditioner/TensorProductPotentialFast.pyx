'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

import pyfftw
# from pyfftw.interfaces.scipy_fftpack import fft, ifft

from petsc4py import PETSc


cdef class TensorProductPreconditionerPotentialFast(TensorProductPreconditionerPotential):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 object da1  not None,
                 object daph not None,
                 Grid   grid not None,
                 Vec    phi  not None):
        '''
        Constructor
        '''
        
        super().__init__(da1, daph, grid, phi)
        
        cdef int xs, xe, ys, ye
        
        # get local x ranges for FFT
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Creating FFTW objects.")
        
        (xs, xe), (ys, ye) = self.day.getRanges()
        
        # FFTW arrays 
        fftw_in   = np.empty((self.grid.nv,      xe-xs), dtype=np.float64,    order='c')
        fftw_out  = np.empty((self.grid.nv//2+1, xe-xs), dtype=np.complex128, order='c')
        ifftw_in  = np.empty((self.grid.nv//2+1, xe-xs), dtype=np.complex128, order='c')
        ifftw_out = np.empty((self.grid.nv,      xe-xs), dtype=np.float64,    order='c')
        
        # create pyFFTW plans
        self.fftw_plan  = pyfftw.FFTW(fftw_in,  fftw_out,  axes=(0,), direction='FFTW_FORWARD',  flags=('FFTW_UNALIGNED','FFTW_DESTROY_INPUT'))
        self.ifftw_plan = pyfftw.FFTW(ifftw_in, ifftw_out, axes=(0,), direction='FFTW_BACKWARD', flags=('FFTW_UNALIGNED','FFTW_DESTROY_INPUT'))
        
        # LAPACK parameters
        self.M = self.grid.nx
        self.N = self.grid.nx
        self.KL = 1
        self.KU = 1
        self.NRHS = 1
        self.LDA  = 4
        self.LDB  = self.grid.nx
        self.T = 'N'
        
        # get local x ranges for solver
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        # build matrices
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Creating Preconditioner Matrices.")
            
        self.matrices = np.zeros((4, xe-xs, ye-ys), dtype=np.cdouble, order='F')
        self.pivots   = np.empty((   xe-xs, ye-ys), dtype=np.int32,   order='F')
        
        self.update_matrices()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Preconditioner Initialisation done.")
            print("")
    
    
#     def __dealloc__(self):
#         del self.fftw_plan
#         del self.ifftw_plan
        
    
    cdef fft(self, Vec X, Vec Y):
        # Fourier Transform for each v
        
        # This code uses some complicated casts to call
        # the pyFFTW wrapper around FFTW.
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.day.getRanges()
           
        cdef np.ndarray[double, ndim=2] x = X.getArray().reshape((ye-ys, xe-xs), order='c')
            
        (xs, xe), (ys, ye) = self.cay.getRanges()

        assert ys == 0
        assert ye == self.grid.nv//2+1
           
        cdef np.ndarray[np.complex128_t, ndim=2] y = np.empty((ye-ys, xe-xs), dtype=np.complex128, order='c')
           
        self.fftw_plan(input_array=x, output_array=y)
   
        (<dcomplex[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(Y.getArray()))[...] = y
         
#         # This code calls directly into FFTW passing the array
#         # buffers of the input and output vectors.
#         # The FFTW plan is still setup using pyFFTW.
#         # Be careful to allow for unaligned input/output arrays.
#         fftw_execute_dft_r2c(<fftw_plan>self.fftw_plan.__plan,
#                              <double*>np.PyArray_DATA(X.getArray()),
#                              <cdouble*>np.PyArray_DATA(Y.getArray()))

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
        
        # This code uses some complicated casts to call
        # the pyFFTW wrapper around FFTW.
        (xs, xe), (ys, ye) = self.day.getRanges()
              
        cdef np.ndarray[double, ndim=2] y = Y.getArray().reshape((ye-ys, xe-xs), order='c')
               
        (xs, xe), (ys, ye) = self.cay.getRanges()
              
        assert ys == 0
        assert ye == self.grid.nv//2+1
           
        cdef np.ndarray[np.complex128_t, ndim=2] x = np.empty((ye-ys, xe-xs), dtype=np.complex128, order='c')
         
        x[...] = (<dcomplex[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(X.getArray()))
              
        self.ifftw_plan(input_array=x, output_array=y)
        
#         # This code calls directly into FFTW passing the array
#         # buffers of the input and output vectors.
#         # The FFTW plan is still setup using pyFFTW.
#         # Be careful to allow for unaligned input/output arrays.
#         fftw_execute_dft_c2r(<fftw_plan>self.ifftw_plan.__plan,
#                              <cdouble*>np.PyArray_DATA(X.getArray()),
#                              <double*>np.PyArray_DATA(Y.getArray()))
#         
#         Y.scale(1./float(self.grid.nv))
        
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
        
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        assert xs == 0
        assert xe == self.grid.nx
        
        cdef dcomplex[:,:] x = <dcomplex[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(X.getArray())
        
        for i in range(ye-ys):
            self.call_zgbtrs(self.matrices[:,:,i], x[i], self.pivots[:,i])
        
    
    cdef call_zgbtrf(self, dcomplex[:,:] matrix, int[:] pivots):
        cdef int INFO = 0
        
        zgbtrf(&self.M, &self.N, &self.KL, &self.KU, &matrix[0,0], &self.LDA, &pivots[0], &INFO)
        
        return INFO
     
 
    cdef call_zgbtrs(self, dcomplex[:,:] matrix, dcomplex[:] rhs, int[:] pivots):
        
        cdef int INFO = 0
         
        zgbtrs(&self.T, &self.N, &self.KL, &self.KU, &self.NRHS, &matrix[0,0], &self.LDA, &pivots[0], &rhs[0], &self.LDB, &INFO)
        
        return INFO
    
    
    cdef update_matrices(self):
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        # build matrices
#         if PETSc.COMM_WORLD.getRank() == 0:
#             print("  Updating Preconditioner Matrices.")
        
        for i in range(ye-ys):
            self.formBandedPreconditionerMatrix(self.matrices[:,:,i], self.eigen[i+ys])
         
        # LU decompositions
#         if PETSc.COMM_WORLD.getRank() == 0:
#             print("  LU Decomposing Preconditioner Matrices.")
         
        for i in range(ye-ys):
            if self.call_zgbtrf(self.matrices[:,:,i], self.pivots[:,i]) != 0:
                print("   ERROR in LU Decomposition.")
    
    
    cdef formBandedPreconditionerMatrix(self, dcomplex[:,:] matrix, dcomplex eigen):
        cdef int j
        
        # get the whole phi array
        scatter, phiVec = PETSc.Scatter.toAll(self.phi)
 
        scatter.begin(self.phi, phiVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (self.phi, phiVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
         
        cdef double[:] phi = phiVec.getValues(range(self.grid.nx)).copy()
         
        scatter.destroy()
        phiVec.destroy()
        
        # Arakawa J1 factor
        cdef double arak_fac_J1 = 0.5 * self.grid.hx_inv * self.grid.hv_inv / 12.
        
        # tri-diagonals of matrix
        cdef dcomplex[:] diagm = np.zeros(self.grid.nx, dtype=np.cdouble)
        cdef dcomplex[:] diag  = np.ones (self.grid.nx, dtype=np.cdouble)
        cdef dcomplex[:] diagp = np.zeros(self.grid.nx, dtype=np.cdouble)
        
        for i in range(self.grid.nx):
            im = (i-1) % self.grid.nx
            ip = (i+1) % self.grid.nx
            
            diagm[i] = eigen * 1.0 * ( phi[i ] - phi[im] ) * arak_fac_J1
            diag [i] = eigen * 2.0 * ( phi[ip] - phi[im] ) * arak_fac_J1 + self.grid.ht_inv
            diagp[i] = eigen * 1.0 * ( phi[ip] - phi[i ] ) * arak_fac_J1

#             diagm[i] = 0.
#             diag [i] = self.grid.ht_inv
#             diagp[i] = 0.
 
#             diagm[i] = 0.
#             diag [i] = 1.
#             diagp[i] = 0.

        matrix[1, 1:  ] = diagp[:-1]
        matrix[2,  :  ] = diag [:]
        matrix[3,  :-1] = diagm[1:]
        
        # TODO Need to update matrices (at least) every timestep.
