'''
Created on Apr 10, 2012

@author: mkraus
'''
cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import  PETSc
from petsc4py cimport PETSc

from petsc4py.PETSc cimport Vec
from petsc4py.PETSc cimport Mat
from petsc4py.PETSc cimport DA   

from cpython cimport bool

#from petsc4py.PETSc cimport NPY_PETSC_INT
#from petsc4py.PETSc cimport IntType
#from petsc4py.PETSc cimport PetscInt
#from petsc4py.PETSc cimport PetscReal


cdef class PETScSolver(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DA da, Vec X, Vec B,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 np.ndarray[np.float64_t, ndim=1] h0,
                 np.float64_t poisson_const):
        '''
        Constructor
        '''
        
        assert da.getDim() == 2
        
        self.eps = 1E-7
        
        # distributed array
        self.da = da
        
        # kinetic Hamiltonian
        self.h0 = h0
        
        # poisson constant
        self.poisson_const = poisson_const
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv
        
        # save solution and RHS vector
        self.X = X
        self.B = B
        
        # create history vectors
        self.X1 = self.da.createGlobalVec()
        self.X2 = self.da.createGlobalVec()
        
        # create local vectors
        self.localB  = da.createLocalVec()
        self.localX  = da.createLocalVec()
        self.localX1 = da.createLocalVec()
        self.localX2 = da.createLocalVec()
        
        # create temporary numpy array
        (xs, xe), (ys, ye) = self.da.getRanges()
        self.ty = np.empty((xe-xs, ye-ys))
        
    
    def update_history(self, Vec X):
#        cdef np.uint64_t xs, xe, ys, ye
        
        x  = self.da.getVecArray(X)
        x1 = self.da.getVecArray(self.X1)
        x2 = self.da.getVecArray(self.X2)
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        x2[xs:xe, ys:ye] = x1[xs:xe, ys:ye]
        x1[xs:xe, ys:ye] = x [xs:xe, ys:ye]
        
#        if self.nmult == 0:
#        viewer = PETSc.Viewer().createDraw()
#        viewer(X)
#        raw_input('Hit any key to continue simulation.')
        
#        viewer = PETSc.Viewer().createASCII('x_%02i.txt' % self.nsolv, 'w')
#        viewer(X)
        
    
    def convergence_test(self, ksp, its, rnorm):
#        self.da.globalToLocal(self.X, self.localX)
#        self.da.globalToLocal(self.B, self.localB)
        
        x = self.da.getVecArray(self.X)
        b = self.da.getVecArray(self.B)

        cdef np.ndarray[np.float64_t, ndim=2] tx = x[...]
        cdef np.ndarray[np.float64_t, ndim=2] tb = b[...]
        
        if np.abs(tx-tb).max() < self.eps:
            return True
        else:
            return False
    
    
#    @cython.boundscheck(False)
    def mult(self, Mat mat, Vec X, Vec Y):
        cdef np.uint64_t i, j, l, lx
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        cdef np.float64_t time_deriv, arakawa_0_1, arakawa_1_0
        cdef np.float64_t laplace, integral
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        y = self.da.getVecArray(Y)
        
        self.da.globalToLocal(X, self.localX)
        self.da.globalToLocal(self.X1, self.localX1)
        self.da.globalToLocal(self.X2, self.localX2)
        
        x  = self.da.getVecArray(self.localX)
        x1 = self.da.getVecArray(self.localX1)
        x2 = self.da.getVecArray(self.localX2)
        
        cdef np.ndarray[np.float64_t, ndim=1] h0  = self.h0
        cdef np.ndarray[np.float64_t, ndim=2] ty  = self.ty
        cdef np.ndarray[np.float64_t, ndim=2] tx  = x[...]
        cdef np.ndarray[np.float64_t, ndim=2] tx1 = x1[...]
        cdef np.ndarray[np.float64_t, ndim=2] tx2 = x2[...]
        
        
        for j in np.arange(ys, ye):
            jx = j-ys+1
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+1
                iy = i-xs
                
                if j == self.nv:
                    # Poisson equation
#                    y[i, j] = 0.0
                    
#                    if i == self.nx-1:
#                        y[i, j] = 0.0
#                        for l in range(0, self.nx):
#                            y[i, j] += x[l, j]
#                    else:
                    laplace  = (tx[ix-1, jx] - 2 * tx[ix, jx] + tx[ix+1, jx]) / self.hx**2
                    integral = 0.0
                    
                    for l in np.arange(0, self.nv):
                        lx = l+1
                        integral -= ( tx[ix-1, lx] + 2 * tx[ix, lx] + tx[ix+1, lx] )
#                            ty[iy, jy] -= ( tgx[ix-1, lx] + 2 * tgx[ix, lx] + tgx[ix+1, lx] ) * self.hv / 4
                    
                    integral  *= 0.25 * self.hv * self.poisson_const
                    ty[iy, jy] = laplace + integral
                    
                else:
                    # Vlasov equation
                    if j > 0 and j < self.nv-1:
                        
                        time_deriv = ( \
                                       + 1 * tx[ix-1, jx-1] \
                                       + 2 * tx[ix,   jx-1] \
                                       + 1 * tx[ix+1, jx-1] \
                                       + 2 * tx[ix-1, jx  ] \
                                       + 4 * tx[ix,   jx  ] \
                                       + 2 * tx[ix+1, jx  ] \
                                       + 1 * tx[ix-1, jx+1] \
                                       + 2 * tx[ix,   jx+1] \
                                       + 1 * tx[ix+1, jx+1] \
                                     ) / (16 * 2 * self.ht)
                        
                        arakawa_0_1 = ( \
                                        + tx[ix-1, jx-1] * (h0[jy  ] + tx1[ix-1, self.nv+1]) \
                                        - tx[ix+1, jx-1] * (h0[jy  ] + tx1[ix+1, self.nv+1]) \
                                        + tx[ix,   jx-1] * (h0[jy-1] + tx1[ix-1, self.nv+1]) \
                                        - tx[ix,   jx-1] * (h0[jy-1] + tx1[ix+1, self.nv+1]) \
                                        - tx[ix-1, jx-1] * (h0[jy-1] + tx1[ix,   self.nv+1]) \
                                        + tx[ix+1, jx-1] * (h0[jy-1] + tx1[ix,   self.nv+1]) \
                                        + tx[ix,   jx-1] * (h0[jy  ] + tx1[ix-1, self.nv+1]) \
                                        - tx[ix,   jx-1] * (h0[jy  ] + tx1[ix+1, self.nv+1]) \
                                        - tx[ix-1, jx  ] * (h0[jy-1] + tx1[ix-1, self.nv+1]) \
                                        - tx[ix-1, jx  ] * (h0[jy-1] + tx1[ix,   self.nv+1]) \
                                        + tx[ix+1, jx  ] * (h0[jy-1] + tx1[ix+1, self.nv+1]) \
                                        + tx[ix+1, jx  ] * (h0[jy-1] + tx1[ix,   self.nv+1]) \
                                        + tx[ix-1, jx  ] * (h0[jy+1] + tx1[ix-1, self.nv+1]) \
                                        + tx[ix-1, jx  ] * (h0[jy+1] + tx1[ix,   self.nv+1]) \
                                        - tx[ix+1, jx  ] * (h0[jy+1] + tx1[ix+1, self.nv+1]) \
                                        - tx[ix+1, jx  ] * (h0[jy+1] + tx1[ix,   self.nv+1]) \
                                        - tx[ix-1, jx+1] * (h0[jy  ] + tx1[ix-1, self.nv+1]) \
                                        + tx[ix+1, jx+1] * (h0[jy  ] + tx1[ix+1, self.nv+1]) \
                                        - tx[ix,   jx+1] * (h0[jy  ] + tx1[ix-1, self.nv+1]) \
                                        + tx[ix,   jx+1] * (h0[jy  ] + tx1[ix+1, self.nv+1]) \
                                        + tx[ix-1, jx+1] * (h0[jy+1] + tx1[ix,   self.nv+1]) \
                                        - tx[ix+1, jx+1] * (h0[jy+1] + tx1[ix,   self.nv+1]) \
                                        - tx[ix,   jx+1] * (h0[jy+1] + tx1[ix-1, self.nv+1]) \
                                        + tx[ix,   jx+1] * (h0[jy+1] + tx1[ix+1, self.nv+1]) \
                                      ) / (4 * 12 * self.hx * self.hv)
                        
#                        arakawa_1_0 = ( \
#                                        + tx1[ix-1, jx-1] * (h0[jy  ] + tx[ix-1, self.nv+1]) \
#                                        - tx1[ix+1, jx-1] * (h0[jy  ] + tx[ix+1, self.nv+1]) \
#                                        + tx1[ix,   jx-1] * (h0[jy-1] + tx[ix-1, self.nv+1]) \
#                                        - tx1[ix,   jx-1] * (h0[jy-1] + tx[ix+1, self.nv+1]) \
#                                        - tx1[ix-1, jx-1] * (h0[jy-1] + tx[ix,   self.nv+1]) \
#                                        + tx1[ix+1, jx-1] * (h0[jy-1] + tx[ix,   self.nv+1]) \
#                                        + tx1[ix,   jx-1] * (h0[jy  ] + tx[ix-1, self.nv+1]) \
#                                        - tx1[ix,   jx-1] * (h0[jy  ] + tx[ix+1, self.nv+1]) \
#                                        - tx1[ix-1, jx  ] * (h0[jy-1] + tx[ix-1, self.nv+1]) \
#                                        - tx1[ix-1, jx  ] * (h0[jy-1] + tx[ix,   self.nv+1]) \
#                                        + tx1[ix+1, jx  ] * (h0[jy-1] + tx[ix+1, self.nv+1]) \
#                                        + tx1[ix+1, jx  ] * (h0[jy-1] + tx[ix,   self.nv+1]) \
#                                        + tx1[ix-1, jx  ] * (h0[jy+1] + tx[ix-1, self.nv+1]) \
#                                        + tx1[ix-1, jx  ] * (h0[jy+1] + tx[ix,   self.nv+1]) \
#                                        - tx1[ix+1, jx  ] * (h0[jy+1] + tx[ix+1, self.nv+1]) \
#                                        - tx1[ix+1, jx  ] * (h0[jy+1] + tx[ix,   self.nv+1]) \
#                                        - tx1[ix-1, jx+1] * (h0[jy  ] + tx[ix-1, self.nv+1]) \
#                                        + tx1[ix+1, jx+1] * (h0[jy  ] + tx[ix+1, self.nv+1]) \
#                                        - tx1[ix,   jx+1] * (h0[jy  ] + tx[ix-1, self.nv+1]) \
#                                        + tx1[ix,   jx+1] * (h0[jy  ] + tx[ix+1, self.nv+1]) \
#                                        + tx1[ix-1, jx+1] * (h0[jy+1] + tx[ix,   self.nv+1]) \
#                                        - tx1[ix+1, jx+1] * (h0[jy+1] + tx[ix,   self.nv+1]) \
#                                        - tx1[ix,   jx+1] * (h0[jy+1] + tx[ix-1, self.nv+1]) \
#                                        + tx1[ix,   jx+1] * (h0[jy+1] + tx[ix+1, self.nv+1]) \
#                                      ) / (4 * 12 * self.hx * self.hv)
                        
                        arakawa_1_0 = ( \
                                        + tx1[ix-1, jx-1] * tx[ix-1, self.nv+1] \
                                        - tx1[ix+1, jx-1] * tx[ix+1, self.nv+1] \
                                        + tx1[ix,   jx-1] * tx[ix-1, self.nv+1] \
                                        - tx1[ix,   jx-1] * tx[ix+1, self.nv+1] \
                                        - tx1[ix-1, jx-1] * tx[ix,   self.nv+1] \
                                        + tx1[ix+1, jx-1] * tx[ix,   self.nv+1] \
                                        + tx1[ix,   jx-1] * tx[ix-1, self.nv+1] \
                                        - tx1[ix,   jx-1] * tx[ix+1, self.nv+1] \
                                        - tx1[ix-1, jx  ] * tx[ix-1, self.nv+1] \
                                        - tx1[ix-1, jx  ] * tx[ix,   self.nv+1] \
                                        + tx1[ix+1, jx  ] * tx[ix+1, self.nv+1] \
                                        + tx1[ix+1, jx  ] * tx[ix,   self.nv+1] \
                                        + tx1[ix-1, jx  ] * tx[ix-1, self.nv+1] \
                                        + tx1[ix-1, jx  ] * tx[ix,   self.nv+1] \
                                        - tx1[ix+1, jx  ] * tx[ix+1, self.nv+1] \
                                        - tx1[ix+1, jx  ] * tx[ix,   self.nv+1] \
                                        - tx1[ix-1, jx+1] * tx[ix-1, self.nv+1] \
                                        + tx1[ix+1, jx+1] * tx[ix+1, self.nv+1] \
                                        - tx1[ix,   jx+1] * tx[ix-1, self.nv+1] \
                                        + tx1[ix,   jx+1] * tx[ix+1, self.nv+1] \
                                        + tx1[ix-1, jx+1] * tx[ix,   self.nv+1] \
                                        - tx1[ix+1, jx+1] * tx[ix,   self.nv+1] \
                                        - tx1[ix,   jx+1] * tx[ix-1, self.nv+1] \
                                        + tx1[ix,   jx+1] * tx[ix+1, self.nv+1] \
                                      ) / (4 * 12 * self.hx * self.hv)
                        
                        ty[iy, jy] = time_deriv - arakawa_0_1 - arakawa_1_0
                    else:
                        # Dirichlet boundary conditions
                        ty[iy, jy] = tx[ix, jx]
        
        y[:,:] = ty[:,:]
        
    
    def formRHS(self, Vec B):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
       
        cdef np.float64_t time_deriv, arakawa_1_2, arakawa_2_1
        
        self.da.globalToLocal(self.X1, self.localX1)
        self.da.globalToLocal(self.X2, self.localX2)
        
        b  = self.da.getVecArray(B)
        x1 = self.da.getVecArray(self.localX1)
        x2 = self.da.getVecArray(self.localX2)
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=1] h0  = self.h0
        cdef np.ndarray[np.float64_t, ndim=2] ty  = self.ty
        cdef np.ndarray[np.float64_t, ndim=2] tx1 = x1[...]
        cdef np.ndarray[np.float64_t, ndim=2] tx2 = x2[...]
        
        
        ty[:,:] = 0.
        
        for j in np.arange(ys, ye):
            jx = j-ys+1
            jy = j-ys
            
            for i in np.arange(xs, xe):
                ix = i-xs+1
                iy = i-xs
                
                if j == self.nv:
                    # Poisson equation
#                    if i == self.nx-1:
#                        b[i, j] = 0.0
#                    else:
#                        b[i, j] = -1.0
                    ty[iy, jy] = - 1.0 * self.poisson_const
                else:
                    # Vlasov equation
                    if j > 0 and j < self.nv-1:
                        
                        time_deriv = ( \
                                       + 1 * tx2[ix-1, jx-1] \
                                       + 1 * tx2[ix+1, jx-1] \
                                       + 2 * tx2[ix,   jx-1] \
                                       + 2 * tx2[ix-1, jx  ] \
                                       + 2 * tx2[ix+1, jx  ] \
                                       + 4 * tx2[ix,   jx  ] \
                                       + 1 * tx2[ix-1, jx+1] \
                                       + 1 * tx2[ix+1, jx+1] \
                                       + 2 * tx2[ix,   jx+1] \
                                      ) / (16 * 2 * self.ht)
                        
#                        arakawa_1_2 = ( \
#                                        + tx1[ix-1, jx-1] * (h0[jy  ] + tx2[ix-1, self.nv+1]) \
#                                        - tx1[ix+1, jx-1] * (h0[jy  ] + tx2[ix+1, self.nv+1]) \
#                                        + tx1[ix,   jx-1] * (h0[jy  ] + tx2[ix-1, self.nv+1]) \
#                                        - tx1[ix,   jx-1] * (h0[jy  ] + tx2[ix+1, self.nv+1]) \
#                                        - tx1[ix-1, jx-1] * (h0[jy-1] + tx2[ix,   self.nv+1]) \
#                                        + tx1[ix+1, jx-1] * (h0[jy-1] + tx2[ix,   self.nv+1]) \
#                                        + tx1[ix,   jx-1] * (h0[jy-1] + tx2[ix-1, self.nv+1]) \
#                                        - tx1[ix,   jx-1] * (h0[jy-1] + tx2[ix+1, self.nv+1]) \
#                                        - tx1[ix-1, jx  ] * (h0[jy-1] + tx2[ix-1, self.nv+1]) \
#                                        - tx1[ix-1, jx  ] * (h0[jy-1] + tx2[ix,   self.nv+1]) \
#                                        + tx1[ix+1, jx  ] * (h0[jy-1] + tx2[ix+1, self.nv+1]) \
#                                        + tx1[ix+1, jx  ] * (h0[jy-1] + tx2[ix,   self.nv+1]) \
#                                        + tx1[ix-1, jx  ] * (h0[jy+1] + tx2[ix-1, self.nv+1]) \
#                                        + tx1[ix-1, jx  ] * (h0[jy+1] + tx2[ix,   self.nv+1]) \
#                                        - tx1[ix+1, jx  ] * (h0[jy+1] + tx2[ix+1, self.nv+1]) \
#                                        - tx1[ix+1, jx  ] * (h0[jy+1] + tx2[ix,   self.nv+1]) \
#                                        + tx1[ix-1, jx+1] * (h0[jy+1] + tx2[ix,   self.nv+1]) \
#                                        - tx1[ix+1, jx+1] * (h0[jy+1] + tx2[ix,   self.nv+1]) \
#                                        - tx1[ix,   jx+1] * (h0[jy+1] + tx2[ix-1, self.nv+1]) \
#                                        + tx1[ix,   jx+1] * (h0[jy+1] + tx2[ix+1, self.nv+1]) \
#                                        - tx1[ix-1, jx+1] * (h0[jy  ] + tx2[ix-1, self.nv+1]) \
#                                        + tx1[ix+1, jx+1] * (h0[jy  ] + tx2[ix+1, self.nv+1]) \
#                                        - tx1[ix,   jx+1] * (h0[jy  ] + tx2[ix-1, self.nv+1]) \
#                                        + tx1[ix,   jx+1] * (h0[jy  ] + tx2[ix+1, self.nv+1]) \
#                                      ) / (4 * 12 * self.hx * self.hv)
                        
                        arakawa_1_2 = ( \
                                        + tx1[ix-1, jx-1] * (2*h0[jy  ] + tx2[ix-1, self.nv+1]) \
                                        - tx1[ix+1, jx-1] * (2*h0[jy  ] + tx2[ix+1, self.nv+1]) \
                                        + tx1[ix,   jx-1] * (2*h0[jy  ] + tx2[ix-1, self.nv+1]) \
                                        - tx1[ix,   jx-1] * (2*h0[jy  ] + tx2[ix+1, self.nv+1]) \
                                        - tx1[ix-1, jx-1] * (2*h0[jy-1] + tx2[ix,   self.nv+1]) \
                                        + tx1[ix+1, jx-1] * (2*h0[jy-1] + tx2[ix,   self.nv+1]) \
                                        + tx1[ix,   jx-1] * (2*h0[jy-1] + tx2[ix-1, self.nv+1]) \
                                        - tx1[ix,   jx-1] * (2*h0[jy-1] + tx2[ix+1, self.nv+1]) \
                                        - tx1[ix-1, jx  ] * (2*h0[jy-1] + tx2[ix-1, self.nv+1]) \
                                        - tx1[ix-1, jx  ] * (2*h0[jy-1] + tx2[ix,   self.nv+1]) \
                                        + tx1[ix+1, jx  ] * (2*h0[jy-1] + tx2[ix+1, self.nv+1]) \
                                        + tx1[ix+1, jx  ] * (2*h0[jy-1] + tx2[ix,   self.nv+1]) \
                                        + tx1[ix-1, jx  ] * (2*h0[jy+1] + tx2[ix-1, self.nv+1]) \
                                        + tx1[ix-1, jx  ] * (2*h0[jy+1] + tx2[ix,   self.nv+1]) \
                                        - tx1[ix+1, jx  ] * (2*h0[jy+1] + tx2[ix+1, self.nv+1]) \
                                        - tx1[ix+1, jx  ] * (2*h0[jy+1] + tx2[ix,   self.nv+1]) \
                                        + tx1[ix-1, jx+1] * (2*h0[jy+1] + tx2[ix,   self.nv+1]) \
                                        - tx1[ix+1, jx+1] * (2*h0[jy+1] + tx2[ix,   self.nv+1]) \
                                        - tx1[ix,   jx+1] * (2*h0[jy+1] + tx2[ix-1, self.nv+1]) \
                                        + tx1[ix,   jx+1] * (2*h0[jy+1] + tx2[ix+1, self.nv+1]) \
                                        - tx1[ix-1, jx+1] * (2*h0[jy  ] + tx2[ix-1, self.nv+1]) \
                                        + tx1[ix+1, jx+1] * (2*h0[jy  ] + tx2[ix+1, self.nv+1]) \
                                        - tx1[ix,   jx+1] * (2*h0[jy  ] + tx2[ix-1, self.nv+1]) \
                                        + tx1[ix,   jx+1] * (2*h0[jy  ] + tx2[ix+1, self.nv+1]) \
                                      ) / (4 * 12 * self.hx * self.hv)
                        
                        arakawa_2_1 = ( \
                                        + tx2[ix-1, jx-1] * (h0[jy  ] + tx1[ix-1, self.nv+1]) \
                                        - tx2[ix+1, jx-1] * (h0[jy  ] + tx1[ix+1, self.nv+1]) \
                                        + tx2[ix,   jx-1] * (h0[jy  ] + tx1[ix-1, self.nv+1]) \
                                        - tx2[ix,   jx-1] * (h0[jy  ] + tx1[ix+1, self.nv+1]) \
                                        - tx2[ix-1, jx-1] * (h0[jy-1] + tx1[ix,   self.nv+1]) \
                                        + tx2[ix+1, jx-1] * (h0[jy-1] + tx1[ix,   self.nv+1]) \
                                        + tx2[ix,   jx-1] * (h0[jy-1] + tx1[ix-1, self.nv+1]) \
                                        - tx2[ix,   jx-1] * (h0[jy-1] + tx1[ix+1, self.nv+1]) \
                                        - tx2[ix-1, jx  ] * (h0[jy-1] + tx1[ix-1, self.nv+1]) \
                                        - tx2[ix-1, jx  ] * (h0[jy-1] + tx1[ix,   self.nv+1]) \
                                        + tx2[ix+1, jx  ] * (h0[jy-1] + tx1[ix+1, self.nv+1]) \
                                        + tx2[ix+1, jx  ] * (h0[jy-1] + tx1[ix,   self.nv+1]) \
                                        + tx2[ix-1, jx  ] * (h0[jy+1] + tx1[ix-1, self.nv+1]) \
                                        + tx2[ix-1, jx  ] * (h0[jy+1] + tx1[ix,   self.nv+1]) \
                                        - tx2[ix+1, jx  ] * (h0[jy+1] + tx1[ix+1, self.nv+1]) \
                                        - tx2[ix+1, jx  ] * (h0[jy+1] + tx1[ix,   self.nv+1]) \
                                        - tx2[ix-1, jx+1] * (h0[jy  ] + tx1[ix-1, self.nv+1]) \
                                        + tx2[ix+1, jx+1] * (h0[jy  ] + tx1[ix+1, self.nv+1]) \
                                        - tx2[ix,   jx+1] * (h0[jy  ] + tx1[ix-1, self.nv+1]) \
                                        + tx2[ix,   jx+1] * (h0[jy  ] + tx1[ix+1, self.nv+1]) \
                                        + tx2[ix-1, jx+1] * (h0[jy+1] + tx1[ix,   self.nv+1]) \
                                        - tx2[ix+1, jx+1] * (h0[jy+1] + tx1[ix,   self.nv+1]) \
                                        - tx2[ix,   jx+1] * (h0[jy+1] + tx1[ix-1, self.nv+1]) \
                                        + tx2[ix,   jx+1] * (h0[jy+1] + tx1[ix+1, self.nv+1]) \
                                      ) / (4 * 12 * self.hx * self.hv)
                        
                        ty[iy, jy] = time_deriv + arakawa_1_2 + arakawa_2_1
                    else:
                        ty[iy, jy] = 0.0
        
        b[:,:] = ty[:,:]
    

    def isSparse(self):
        return False
    
