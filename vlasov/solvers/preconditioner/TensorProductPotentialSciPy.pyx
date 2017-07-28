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


cdef class TensorProductPreconditionerPotentialSciPy(TensorProductPreconditionerPotential):
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
                
        # get local x ranges for solver
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        # matrices, rhs, pivots
        self.matrices = np.zeros((3, xe-xs, ye-ys), dtype=np.cdouble, order='F')
        
        # build matrices
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Creating Preconditioner Matrices.")
            
        self.update_matrices()
        
        
    cdef fft(self, Vec X, Vec Y):
        # Fourier Transform for each v
        
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.day.getRanges()
        
        cdef double[:,:] x = <double[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(X.getArray())
        
        (xs, xe) , (ys, ye) = self.cay.getRanges()
        
        assert ys == 0
        assert ye == self.grid.nv//2+1
        
        cdef dcomplex[:,:] y = <dcomplex[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(Y.getArray())
        
        np.asarray(y)[...] = rfft(np.asarray(x), axis=0)

    
    cdef ifft(self, Vec X, Vec Y):
        # inverse Fourier Transform for each v
        
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.day.getRanges()
        
        cdef double[:,:] y = <double[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(Y.getArray())
        
        (xs, xe), (ys, ye) = self.cay.getRanges()
        
        assert ys == 0
        assert ye == self.grid.nv//2+1
        
        cdef dcomplex[:,:] x = <dcomplex[:(ye-ys), :(xe-xs)]> np.PyArray_DATA(X.getArray())
        
        np.asarray(y)[...] = irfft(np.asarray(x), axis=0)
    
    
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
#             np.asarray(x[i])[...] = self.solvers[i].solve(np.asarray(x[i]))
            np.asarray(x[i])[...] = solve_banded((1,1), self.matrices[:,:,i], np.asarray(x[i]), overwrite_b=True)
        
    
    cdef update_matrices(self):
        cdef int i, xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.cax.getRanges()
        
        for i in range(ye-ys):
            self.formBandedPreconditionerMatrix(self.matrices[:,:,i], self.eigen[i+ys])
         
        # LU decompositions
#         self.solvers = [splu(self.formSparsePreconditionerMatrix(eigen[i+xs])) for i in range(0, xe-xs)]
    

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
        cdef dcomplex[:] diagm = np.zeros(self.grid.nx, dtype=np.complex128)
        cdef dcomplex[:] diag  = np.ones (self.grid.nx, dtype=np.complex128)
        cdef dcomplex[:] diagp = np.zeros(self.grid.nx, dtype=np.complex128)
        
        for i in range(self.grid.nx):
            im = (i-1) % self.grid.nx
            ip = (i+1) % self.grid.nx
             
            diagm[i] = eigen * 1.0 * ( phi[i ] - phi[im] ) * arak_fac_J1
            diag [i] = eigen * 2.0 * ( phi[ip] - phi[im] ) * arak_fac_J1 + self.grid.ht_inv
            diagp[i] = eigen * 1.0 * ( phi[ip] - phi[i ] ) * arak_fac_J1
            
#             diagm[i] = 0.
#             diag [i] = self.grid.ht_inv
#             diagp[i] = 0.

        matrix[0, 1:  ] = diagp[:-1]
        matrix[1,  :  ] = diag [:]
        matrix[2,  :-1] = diagm[1:]
        
#         offsets   = [-1, 0, +1]
#         diagonals = [diagm[1:], diag, diagp[:-1]]
        
#         return diags(diagonals, offsets, shape=(self.grid.nv, self.grid.nv), format='csc', dtype=np.complex128)
        
        # TODO Need to update matrices (at least) every timestep.
