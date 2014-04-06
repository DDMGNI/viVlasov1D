'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython
from cython cimport view

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
        
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        cdef double[:,:] x = <double[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(X.getArray())
        
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        cdef dcomplex[:,:] y = <dcomplex[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(Y.getArray())
        
        assert xs == 0
        assert xe == self.grid.nx//2+1
        
        np.asarray(y)[...] = rfft(np.asarray(x), axis=1)

    
    cdef ifft(self, Vec X, Vec Y):
        # inverse Fourier Transform for each v
        
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        cdef double[:,:] y = <double[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(Y.getArray())
        
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        cdef dcomplex[:,:] x = <dcomplex[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(X.getArray())
        
        assert xs == 0
        assert xe == self.grid.nx//2+1
        
        np.asarray(y)[...] = irfft(np.asarray(x), axis=1)
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef solve(self, Vec X):
        # solve system for each x
        
        cdef int i, j
        cdef int xe, xs, ye, ys
        
        (ys, ye), (xs, xe) = self.cay.getRanges()
        
        assert ys == 0
        assert ye == self.grid.nv
        
        cdef dcomplex[:,:] x = <dcomplex[:(xe-xs), :(ye-ys)]> np.PyArray_DATA(X.getArray())
        
        for i in range(0, xe-xs):
            np.asarray(x[i])[...] = self.solvers[i].solve(np.asarray(x[i]))
        
    

    cdef formSparsePreconditionerMatrix(self, np.complex eigen):
        
        cdef int j
        
        cdef double arak_fac_J1 = 0.5 * self.grid.hx_inv * self.grid.hv_inv / 12.
        
        diagm = np.zeros(self.grid.nv, dtype=np.complex128)
        diag  = np.ones (self.grid.nv, dtype=np.complex128)
        diagp = np.zeros(self.grid.nv, dtype=np.complex128)
        
        for j in range(2, self.grid.nv-2):
            diagm[j] = eigen * 0.5 * ( 2. * self.grid.hv * self.grid.v[j] - self.grid.hv2 ) * arak_fac_J1
            diag [j] = eigen * 1.0 * ( 4. * self.grid.hv * self.grid.v[j]                 ) * arak_fac_J1 + self.grid.ht_inv
            diagp[j] = eigen * 0.5 * ( 2. * self.grid.hv * self.grid.v[j] + self.grid.hv2 ) * arak_fac_J1
        
        offsets   = [-1, 0, +1]
        diagonals = [diagm[1:], diag, diagp[:-1]]
        
        return diags(diagonals, offsets, shape=(self.grid.nv, self.grid.nv), format='csc', dtype=np.complex128)
        
