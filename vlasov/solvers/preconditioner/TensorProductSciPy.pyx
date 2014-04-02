'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from scipy.sparse        import diags
from scipy.sparse.linalg import splu
from numpy.fft           import rfft, irfft, fftshift, ifftshift

from petsc4py import PETSc


cdef class TensorProductPreconditionerSciPy(TensorProductPreconditioner):
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
        
        # get local x ranges for solver
        (xs, xe), (ys, ye) = self.day.getRanges()
        
        # eigenvalues
        eigen = np.empty(self.grid.nx, dtype=np.complex128)
        
        for i in range(0, self.grid.nx):
            eigen[i] = np.exp(2.j * np.pi * float(i) / self.grid.nx * (self.grid.nx-1)) \
                     - np.exp(2.j * np.pi * float(i) / self.grid.nx)
        
        eigen[:] = ifftshift(eigen)
        
        # LU decompositions
        self.solvers = [splu(self.formSparsePreconditionerMatrix(eigen[i+xs])) for i in range(0, xe-xs)]
        
        
    cdef fft(self, Vec X, Vec Y):
        # Fourier Transform for each v
        
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        dshape = (ye-ys, xe-xs)
        
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        assert xs == 0
        assert xe == self.grid.nx//2+1
        
        cdef np.ndarray[double,   ndim=2] x = X.getArray().reshape(dshape, order='c')
        
        print("fft in")
        
        cdef dcomplex    *yp = <dcomplex *> np.PyArray_DATA(Y.getArray())
        cdef dcomplex[:,:] y = <dcomplex[:ye-ys, :xe-xs]> yp
        
#         cdef np.ndarray[dcomplex, ndim=2] y
#         cdef np.ndarray[dcomplex, ndim=2] y = <dcomplex[:(ye-ys),:(xe-xs)]>(np.PyArray_DATA(Y.getArray()))
        
#         cdef np.ndarray[dcomplex, ndim=2] y = <np.ndarray[dcomplex, ndim=2, mode="c"]>(np.PyArray_DATA(Y.getArray()))
#         cdef np.ndarray[dcomplex, ndim=2] y = <double complex[:(ye-ys),:(xe-xs)]>(np.PyArray_DATA(Y.getArray()))
        
        print("do fft")
        
        y[:,:] = rfft(x, axis=1)
#         z[...] = rfft(x, axis=1)
        
        print("fft out")
        
        
#         (<dcomplex[:ye-ys, :xe-xs]> np.PyArray_DATA(Y.getArray()))[...] = y
        
    
    cdef ifft(self, Vec X, Vec Y):
        # inverse Fourier Transform for each v
        
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        dshape = (ye-ys, xe-xs)
        
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        assert xs == 0
        assert xe == self.grid.nx//2+1
        
        cdef np.ndarray[double,   ndim=2] y = Y.getArray().reshape(dshape, order='c')
        cdef np.ndarray[dcomplex, ndim=2] x = np.empty(((ye-ys),(xe-xs)), dtype=np.complex128) 
#         x[...] = (<dcomplex[:(ye-ys),:(xe-xs)]> np.PyArray_DATA(X.getArray()))
        
        print("ifft in")
        
        x[...] = (<np.ndarray[dcomplex, ndim=2, mode="c"]>(np.PyArray_DATA(X.getArray())))[:(xe-xs),:(ye-ys)]
        
        print("do ifft")
        
        y[:,:] = irfft(x, axis=1)
        
        print("ifft out")
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef solve(self, Vec X):
        # solve system for each x
        
        cdef int i, j
        cdef int xe, xs, ye, ys
        
        (ys, ye), (xs, xe) = self.cay.getRanges()
        
        assert ys == 0
        assert ye == self.grid.nv
        
        print("solve in")
        
        cdef np.ndarray[np.complex128_t, ndim=2] y = np.empty(((xe-xs),(ye-ys)), dtype=np.complex128)
        cdef np.ndarray[np.complex128_t, ndim=2] x = np.empty(((xe-xs),(ye-ys)), dtype=np.complex128) 
#         x[...] = (<dcomplex[:(xe-xs),:(ye-ys)]> np.PyArray_DATA(X.getArray()))
        x[...] = (<np.ndarray[dcomplex, ndim=2, mode="c"]>(np.PyArray_DATA(X.getArray())))[:(xe-xs),:(ye-ys)]
        
        for i in range(0, xe-xs):
            y[i,:] = self.solvers[i].solve(x[i,:])
            
#         (<dcomplex[:(xe-xs),:(ye-ys)]> np.PyArray_DATA(X.getArray()))[...] = y
        x = (<np.ndarray[dcomplex, ndim=2, mode="c"]>(np.PyArray_DATA(X.getArray())))[:(xe-xs),:(ye-ys)]
        x[...] = y
        
        print("solve out")
        

    cdef formSparsePreconditionerMatrix(self, np.complex eigen):
        cdef int j
        
        cdef np.ndarray[double, ndim=1] v = self.grid.v
        
        cdef double arak_fac_J1 = 0.5 / (12. * self.grid.hx * self.grid.hv)
        
        diagm = np.zeros(self.grid.nv, dtype=np.complex128)
        diag  = np.ones (self.grid.nv, dtype=np.complex128)
        diagp = np.zeros(self.grid.nv, dtype=np.complex128)
        
        for j in range(2, self.grid.nv-2):
            diagm[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] - self.grid.hv2 ) * arak_fac_J1
            diag [j] = eigen * 4.0 * self.grid.hv * v[j] * arak_fac_J1 + self.grid.ht_inv
            diagp[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] + self.grid.hv2 ) * arak_fac_J1
        
        offsets   = [-1, 0, +1]
        diagonals = [diagm[1:], diag, diagp[:-1]]
        
        return diags(diagonals, offsets, shape=(self.grid.nv, self.grid.nv), format='csc', dtype=np.complex128)
        
