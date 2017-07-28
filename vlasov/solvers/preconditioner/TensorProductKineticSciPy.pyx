'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from scipy.sparse        import diags
from scipy.sparse.linalg import splu
from scipy.linalg        import solve_banded
from numpy.fft           import rfft, irfft

from petsc4py import PETSc


cdef class TensorProductPreconditionerKineticSciPy(TensorProductPreconditionerKinetic):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 object da1  not None,
                 Grid   grid not None):
        '''
        Constructor
        '''
        
        super().__init__(da1, grid)
        
        # get local x ranges for solver
        (ys, ye), (xs, xe) = self.cay.getRanges()
        
        # matrices, rhs, pivots
        self.matrices = np.zeros((3, ye-ys, xe-xs), dtype=np.cdouble, order='F')
        
        # build matrices
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Creating Preconditioner Matrices.")
        
        for i in range(xe-xs):
            self.formBandedPreconditionerMatrix(self.matrices[:,:,i], self.eigen[i+xs])
         
        # LU decompositions
#         self.solvers = [splu(self.formSparsePreconditionerMatrix(eigen[i+xs])) for i in range(0, xe-xs)]
        
        
    cdef fft(self, Vec X, Vec Y):
        # Fourier Transform for each v
        
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        cdef double[:,:] x = <double[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(X.getArray())
        
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        assert xs == 0
        assert xe == self.grid.nx//2+1
        
        cdef dcomplex[:,:] y = <dcomplex[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(Y.getArray())
        
        np.asarray(y)[...] = rfft(np.asarray(x), axis=1)

    
    cdef ifft(self, Vec X, Vec Y):
        # inverse Fourier Transform for each v
        
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        cdef double[:,:] y = <double[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(Y.getArray())
        
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        assert xs == 0
        assert xe == self.grid.nx//2+1
        
        cdef dcomplex[:,:] x = <dcomplex[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(X.getArray())
        
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
        
        for i in range(xe-xs):
#             np.asarray(x[i])[...] = self.solvers[i].solve(np.asarray(x[i]))
            np.asarray(x[i])[...] = solve_banded((1,1), self.matrices[:,:,i], np.asarray(x[i]), overwrite_b=True)
        
    
    cdef update_matrices(self):
        pass


    cdef formBandedPreconditionerMatrix(self, dcomplex[:,:] matrix, dcomplex eigen):
        cdef int j
        
#        cdef double[:] v = self.grid.v
        
        cdef double arak_fac_J1 = 0.5 * self.grid.hx_inv * self.grid.hv_inv / 12.
        
        cdef dcomplex[:] diagm = np.zeros(self.grid.nv, dtype=np.complex128)
        cdef dcomplex[:] diag  = np.ones (self.grid.nv, dtype=np.complex128)
        cdef dcomplex[:] diagp = np.zeros(self.grid.nv, dtype=np.complex128)
        
        for j in range(self.grid.nv):
            diagm[j] = eigen * 0.5 * ( 2. * self.grid.hv * self.grid.v[j] - self.grid.hv2 ) * arak_fac_J1
            diag [j] = eigen * 1.0 * ( 4. * self.grid.hv * self.grid.v[j]                 ) * arak_fac_J1 + self.grid.ht_inv
            diagp[j] = eigen * 0.5 * ( 2. * self.grid.hv * self.grid.v[j] + self.grid.hv2 ) * arak_fac_J1
         
#             diagm[j] = 0.
#             diag [j] = self.grid.ht_inv
#             diagp[j] = 0.
        
        matrix[0, 1:  ] = diagp[:-1]
        matrix[1,  :  ] = diag [:]
        matrix[2,  :-1] = diagm[1:]
        
#         offsets   = [-1, 0, +1]
#         diagonals = [diagm[1:], diag, diagp[:-1]]
        
#         return diags(diagonals, offsets, shape=(self.grid.nv, self.grid.nv), format='csc', dtype=np.complex128)
        
