# cython: profile=True
'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython
import pyfftw

import  numpy as npy
cimport numpy as npy

from scipy.sparse        import diags, eye
from scipy.sparse.linalg import splu
from scipy.fftpack       import fftshift, ifftshift

from pyfftw.interfaces.scipy_fftpack import fft, ifft

from petsc4py import PETSc


cdef class PETScVlasovSolver(PETScVlasovPreconditioner):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 VIDA da1  not None,
                 Grid grid not None,
                 Vec H0  not None,
                 Vec H1p not None,
                 Vec H1h not None,
                 Vec H2p not None,
                 Vec H2h not None,
                 npy.float64_t charge=-1.,
                 npy.float64_t coll_freq=0.,
                 npy.float64_t coll_diff=1.,
                 npy.float64_t coll_drag=1.,
                 npy.float64_t regularisation=0.):
        '''
        Constructor
        '''
        
        super().__init__(da1, grid, H0, H1p, H1h, H2p, H2h, charge, coll_freq, coll_diff, coll_drag, regularisation)
        
        cdef int i, j, xs, xe, ys, ye
        
        # get local x ranges for solver
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # get local x ranges for FFT
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        # FFTW arrays 
        fftw_in   = npy.empty((ye-ys, self.grid.nx),          dtype=npy.float64, order='c')
        fftw_out  = npy.empty((ye-ys, int(self.grid.nx/2)+1), dtype=npy.cdouble, order='c')
        ifftw_in  = npy.empty((ye-ys, int(self.grid.nx/2)+1), dtype=npy.cdouble, order='c')
        ifftw_out = npy.empty((ye-ys, self.grid.nx),          dtype=npy.float64, order='c')
        
        # enable cache in pyFFTW for optimal performance
        pyfftw.interfaces.cache.enable()
        
        # create pyFFTW plans
        self.fftw_plan  = pyfftw.FFTW(fftw_in,  fftw_out,  axes=(1,), direction='FFTW_FORWARD',  flags=('FFTW_UNALIGNED',))
        self.ifftw_plan = pyfftw.FFTW(ifftw_in, ifftw_out, axes=(1,), direction='FFTW_BACKWARD', flags=('FFTW_UNALIGNED',))
        
        
        # eigenvalues
        eigen = npy.empty(self.grid.nx, dtype=npy.complex128)
        
        for i in range(0, self.grid.nx):
            eigen[i] = npy.exp(2.j * npy.pi * float(i) / self.grid.nx * (self.grid.nx-1)) \
                     - npy.exp(2.j * npy.pi * float(i) / self.grid.nx)
        
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
        (xs, xe), (ys, ye) = self.cay.getRanges()
        
        # matrices, rhs, pivots
        self.matrices = npy.zeros((4, self.grid.nv, xe-xs), dtype=npy.cdouble, order='F')
        self.rhs_arr  = npy.empty((1, self.grid.nv, xe-xs), dtype=npy.cdouble, order='F')
        self.rhs      = self.rhs_arr
        self.pivots   = npy.empty((self.grid.nv, xe-xs), dtype=npy.int32, order='F')
        
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
#         cdef npy.ndarray[npy.complex128_t, ndim=2] y = npy.empty((ye-ys, int(self.grid.nx/2)+1), dtype=npy.complex128, order='c')
#         cdef npy.ndarray[npy.float64_t, ndim=2] x = X.getArray().reshape(dshape, order='c')
#          
#         self.fftw_plan(input_array=x, output_array=y)
#  
#         cdef npy.ndarray[npy.float64_t,    ndim=1] ty = Y.getArray()
#         (<dcomplex[:(ye-ys),:(xe-xs)]> npy.PyArray_DATA(ty))[...] = self.fftw_out
        
        # This code calls directly into FFTW passing the array
        # buffers of the input and output vectors.
        # The FFTW plan is still setup using pyFFTW.
        # Be careful to allow for unaligned input/output arrays.
        fftw_execute_dft_r2c(<fftw_plan>self.fftw_plan.__plan,
                             <double*>npy.PyArray_DATA(X.getArray()),
                             <double complex*>npy.PyArray_DATA(Y.getArray()))
        

    
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
#         cdef npy.ndarray[npy.float64_t,    ndim=1] tx = X.getArray()
#         cdef npy.ndarray[npy.complex128_t, ndim=2] x  = npy.empty(((ye-ys),(xe-xs)), dtype=npy.complex128) 
#         x[...] = (<dcomplex[:(ye-ys),:(xe-xs)]> npy.PyArray_DATA(tx))
#          
#         cdef npy.ndarray[npy.float64_t, ndim=2]    y = Y.getArray().reshape(dshape, order='c')
#             
#         self.ifftw_plan(input_array=x, output_array=y)
        
        # This code calls directly into FFTW passing the array
        # buffers of the input and output vectors.
        # The FFTW plan is still setup using pyFFTW.
        # Be careful to allow for unaligned input/output arrays.
        fftw_execute_dft_c2r(<fftw_plan>self.ifftw_plan.__plan,
                             <double complex*>npy.PyArray_DATA(X.getArray()),
                             <double*>npy.PyArray_DATA(Y.getArray()))
        
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef solve(self, Vec X, Vec Y):
        # solve system for each x
        
        cdef npy.int64_t i, j
        cdef npy.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.cay.getRanges()
        
        assert ys == 0
        assert ye == self.grid.nv
        
        shape = (ye-ys, xe-xs)
        
        cdef npy.ndarray[npy.float64_t,    ndim=1] tx = X.getArray()
        cdef npy.ndarray[npy.float64_t,    ndim=1] ty = Y.getArray()
         
        cdef dcomplex[:,:] cx = (<dcomplex[:(ye-ys),:(xe-xs)]> npy.PyArray_DATA(tx))
        
        cdef npy.ndarray[npy.complex128_t, ndim=3] x_arr = self.rhs_arr
        
        cdef dcomplex[:,:] x = x_arr[0,:,:] 
        x[...] = cx[...]
        
        for i in range(0, xe-xs):
            self.call_zgbtrs(self.matrices[:,:,i], x[:,i], self.pivots[:,i])
        
        (<dcomplex[:(ye-ys),:(xe-xs)]> npy.PyArray_DATA(ty))[:] = self.rhs_arr[0,:,:]
#         ty[...] = tx[...]
        
    
    cdef call_zgbtrf(self, dcomplex[:,:] matrix, int[:] pivots):
        cdef int INFO = 0
          
        zgbtrf(&self.M, &self.N, &self.KL, &self.KU, &matrix[0,0], &self.LDA, &pivots[0], &INFO)
        
        return INFO
     
 
    cdef call_zgbtrs(self, dcomplex[:,:] matrix, dcomplex[:] rhs, int[:] pivots):
        
        cdef int INFO = 0
         
        zgbtrs(&self.T, &self.N, &self.KL, &self.KU, &self.NRHS, &matrix[0,0], &self.LDA, &pivots[0], &rhs[0], &self.LDB, &INFO)
        
        return INFO
        
    
    cdef formBandedPreconditionerMatrix(self, dcomplex[:,:] matrix, npy.complex eigen):
        cdef npy.int64_t j
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v = self.grid.v
        
        cdef npy.float64_t arak_fac_J1 = 0.5 / (12. * self.grid.hx * self.grid.hv)
        
        
        cdef dcomplex[:] diagm = npy.zeros(self.grid.nv, dtype=npy.cdouble)
        cdef dcomplex[:] diag  = npy.ones (self.grid.nv, dtype=npy.cdouble)
        cdef dcomplex[:] diagp = npy.zeros(self.grid.nv, dtype=npy.cdouble)
        
        for j in range(2, self.grid.nv-2):
            diagm[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] - self.grid.hv2 ) * arak_fac_J1
            diag [j] = eigen * 4.0 * self.grid.hv * v[j] * arak_fac_J1 + self.grid.ht_inv
            diagp[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] + self.grid.hv2 ) * arak_fac_J1
        
        matrix[1, 1:  ] = diagp[:-1]
        matrix[2,  :  ] = diag[:]
        matrix[3,  :-1] = diagm[1:]
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef jacobianSolver(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4, poisson
        cdef npy.float64_t coll_drag, coll_diff
        cdef npy.float64_t collisions     = 0.
        cdef npy.float64_t regularisation = 0.
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double[:,:] fd    = self.da1.getLocalArray(F, self.localFd)
        cdef double[:,:] y     = self.da1.getGlobalArray(Y)
        cdef double[:,:] h_ave = self.da1.getLocalArray(self.Have, self.localHave)
        
        cdef double[:] v = self.grid.v
        cdef double[:] u = self.Up.getArray()
        cdef double[:] a = self.Ap.getArray()
        
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy] = fd[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    # Arakawa's J1
                    jpp_J1 = (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                           - (fd[ix,   jx+1] - fd[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                    
                    jpc_J1 = fd[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - fd[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - fd[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + fd[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J1 = fd[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                           - fd[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                           - fd[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                           + fd[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                    
                    # Arakawa's J2
                    jcc_J2 = (fd[ix+1, jx+1] - fd[ix-1, jx-1]) * (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) \
                           - (fd[ix-1, jx+1] - fd[ix+1, jx-1]) * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1])
                    
                    jpc_J2 = fd[ix+2, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - fd[ix-2, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - fd[ix,   jx+2] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + fd[ix,   jx-2] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J2 = fd[ix+1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) \
                           - fd[ix-1, jx-1] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx-2]) \
                           - fd[ix-1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix-2, jx  ]) \
                           + fd[ix+1, jx-1] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    poisson   = 0.5 * result_J4 * self.grid.hx_inv * self.grid.hv_inv \
                    
                    
                    # collision operator
                    if self.nu > 0.:
                        coll_drag = ( (v[j+1] - u[ix]) * fd[ix, jx+1] - (v[j-1] - u[ix]) * fd[ix, jx-1] ) * a[ix]
                        
                        coll_diff = ( fd[ix, jx+1] - 2. * fd[ix, jx] + fd[ix, jx-1] )
                        
                        collisions = \
                              - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                              - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv \
                    
                    # regularisation
                    if self.regularisation != 0.:
                        regularisation = \
                              + self.grid.ht * self.regularisation * self.grid.hx2_inv * ( 2. * fd[ix, jx] - fd[ix+1, jx] - fd[ix-1, jx] ) \
                              + self.grid.ht * self.regularisation * self.grid.hv2_inv * ( 2. * fd[ix, jx] - fd[ix, jx+1] - fd[ix, jx-1] )
                    
                    # solution
                    y[iy, jy] = fd[ix, jx] * self.grid.ht_inv \
                              + poisson \
                              + collisions \
                              + regularisation
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef functionSolver(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4, poisson
        cdef npy.float64_t coll_drag, coll_diff
        cdef npy.float64_t collisions = 0.
        cdef npy.float64_t regularisation = 0.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.Fave.set(0.)
        self.Fave.axpy(0.5, self.Fh)
        self.Fave.axpy(0.5, F)
        
        cdef double[:,:] y     = self.da1.getGlobalArray(Y)
        cdef double[:,:] fp    = self.da1.getLocalArray(F, self.localFp)
        cdef double[:,:] fh    = self.da1.getLocalArray(self.Fh, self.localFh)
        cdef double[:,:] f_ave = self.da1.getLocalArray(self.Fave, self.localFave)
        cdef double[:,:] h_ave = self.da1.getLocalArray(self.Have, self.localHave)
        
        cdef double[:] v  = self.grid.v
        cdef double[:] up = self.Up.getArray()
        cdef double[:] uh = self.Uh.getArray()
        cdef double[:] ap = self.Ap.getArray()
        cdef double[:] ah = self.Ah.getArray()
        
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy] = fp[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    # Arakawa's J1
                    jpp_J1 = (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                           - (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                    
                    jpc_J1 = f_ave[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - f_ave[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - f_ave[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + f_ave[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J1 = f_ave[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                           - f_ave[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                           - f_ave[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                           + f_ave[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                    
                    # Arakawa's J2
                    jcc_J2 = (f_ave[ix+1, jx+1] - f_ave[ix-1, jx-1]) * (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) \
                           - (f_ave[ix-1, jx+1] - f_ave[ix+1, jx-1]) * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1])
                    
                    jpc_J2 = f_ave[ix+2, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - f_ave[ix-2, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - f_ave[ix,   jx+2] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + f_ave[ix,   jx-2] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J2 = f_ave[ix+1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) \
                           - f_ave[ix-1, jx-1] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx-2]) \
                           - f_ave[ix-1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix-2, jx  ]) \
                           + f_ave[ix+1, jx-1] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    poisson   = result_J4 * self.grid.hx_inv * self.grid.hv_inv
                    
                    
                    # collision operator
                    if self.nu > 0.:
                        coll_drag = ( (v[j+1] - up[ix]) * fp[ix, jx+1] - (v[j-1] - up[ix]) * fp[ix, jx-1] ) * ap[ix] \
                                  + ( (v[j+1] - uh[ix]) * fh[ix, jx+1] - (v[j-1] - uh[ix]) * fh[ix, jx-1] ) * ah[ix]
                        
                        coll_diff = ( fp[ix, jx+1] - 2. * fp[ix, jx] + fp[ix, jx-1] ) \
                                  + ( fh[ix, jx+1] - 2. * fh[ix, jx] + fh[ix, jx-1] )
                        
                        collisions = \
                                   - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                                   - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv \
                    
                    # regularisation
                    if self.regularisation != 0.:
                        regularisation = \
                                       + self.grid.ht * self.regularisation * self.grid.hx2_inv * ( 2. * fp[ix, jx] - fp[ix+1, jx] - fp[ix-1, jx] ) \
                                       + self.grid.ht * self.regularisation * self.grid.hv2_inv * ( 2. * fp[ix, jx] - fp[ix, jx+1] - fp[ix, jx-1] )
                    
                    # solution
                    y[iy, jy] = (fp[ix, jx] - fh[ix, jx]) * self.grid.ht_inv \
                              + poisson \
                              + collisions \
                              + regularisation
