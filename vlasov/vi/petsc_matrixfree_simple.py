'''
Created on Apr 10, 2012

@author: mkraus
'''

import  numpy as np

from petsc4py import  PETSc


class PETScSolver(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    
    def __init__(self, da, X, B,
                 nx, nv, ht, hx, hv,
                 h0, poisson_const):
        '''
        Constructor
        '''
        
        self.sparse = False
        
        assert da.getDim() == 2
        
        self.eps = 1E-7
        
        # disstributed array
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
        
        # create local vectors
        self.localB  = da.createLocalVec()
        self.localX  = da.createLocalVec()
        self.localX1 = da.createLocalVec()
        
        # create temporary numpy array
        (xs, xe), (ys, ye) = self.da.getRanges()
        self.ty = np.empty((xe-xs, ye-ys))
        
    
    def update_history(self, X):
        
        x  = self.da.getVecArray(X)
        x1 = self.da.getVecArray(self.X1)
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        x1[xs:xe, ys:ye] = x[xs:xe, ys:ye]
        
    
    def convergence_test(self, ksp, its, rnorm):
        
        x = self.da.getVecArray(self.X)
        b = self.da.getVecArray(self.B)

        if np.abs(x-b).max() < self.eps:
            return True
        else:
            return False
    
    
    def mult(self, mat, X, Y):
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        self.da.globalToLocal(X, self.localX)
        self.da.globalToLocal(self.X1, self.localX1)
        
        y  = self.da.getVecArray(Y)
        x  = self.da.getVecArray(self.localX)
        x1 = self.da.getVecArray(self.localX1)
        h0 = self.h0
        
        
        for j in np.arange(ys, ye):
            for i in np.arange(xs, xe):
                
                if j == self.nv:
                    # Poisson equation
                    
                    laplace  = (x[i-1, j] - 2 * x[i, j] + x[i+1, j]) / self.hx**2
                    
                    integral = ( \
                                 + 1. * x[i-1, 0:self.nv].sum() \
                                 + 2. * x[i,   0:self.nv].sum() \
                                 + 1. * x[i+1, 0:self.nv].sum() \
                               ) * 0.25 * self.hv
                    
                    y[i, j] = laplace - self.poisson_const * integral
                    
                elif j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    y[i, j] = x[i, j]
                    
                else:
                    # Vlasov equation
                    time_deriv = ( \
                                   + 1. * x[i-1, j-1] \
                                   + 2. * x[i,   j-1] \
                                   + 1. * x[i+1, j-1] \
                                   + 2. * x[i-1, j  ] \
                                   + 4. * x[i,   j  ] \
                                   + 2. * x[i+1, j  ] \
                                   + 1. * x[i-1, j+1] \
                                   + 2. * x[i,   j+1] \
                                   + 1. * x[i+1, j+1] \
                                 ) / (16. * self.ht)
                    
                    arakawa_0_h0 = ( \
                                     + x[i-1, j-1] * h0[j  ] \
                                     - x[i-1, j-1] * h0[j-1] \
                                     - x[i-1, j+1] * h0[j  ] \
                                     + x[i-1, j+1] * h0[j+1] \
                                     - x[i-1, j  ] * h0[j-1] \
                                     + x[i-1, j  ] * h0[j+1] \
                                     - x[i-1, j  ] * h0[j-1] \
                                     + x[i-1, j  ] * h0[j+1] \
                                     - x[i+1, j-1] * h0[j  ] \
                                     + x[i+1, j-1] * h0[j-1] \
                                     + x[i+1, j+1] * h0[j  ] \
                                     - x[i+1, j+1] * h0[j+1] \
                                     + x[i+1, j  ] * h0[j-1] \
                                     - x[i+1, j  ] * h0[j+1] \
                                     + x[i+1, j  ] * h0[j-1] \
                                     - x[i+1, j  ] * h0[j+1] \
                                     + x[i,   j-1] * h0[j-1] \
                                     + x[i,   j-1] * h0[j  ] \
                                     - x[i,   j-1] * h0[j-1] \
                                     - x[i,   j-1] * h0[j  ] \
                                     - x[i,   j+1] * h0[j+1] \
                                     - x[i,   j+1] * h0[j  ] \
                                     + x[i,   j+1] * h0[j+1] \
                                     + x[i,   j+1] * h0[j  ] \
                                   ) / (2. * 12. * self.hx * self.hv)
                    
                    arakawa_0_1 = ( \
                                    + x[i-1, j-1] * x1[i-1, self.nv] \
                                    - x[i+1, j-1] * x1[i+1, self.nv] \
                                    + x[i,   j-1] * x1[i-1, self.nv] \
                                    - x[i,   j-1] * x1[i+1, self.nv] \
                                    - x[i-1, j-1] * x1[i,   self.nv] \
                                    + x[i+1, j-1] * x1[i,   self.nv] \
                                    + x[i,   j-1] * x1[i-1, self.nv] \
                                    - x[i,   j-1] * x1[i+1, self.nv] \
                                    - x[i-1, j  ] * x1[i-1, self.nv] \
                                    - x[i-1, j  ] * x1[i,   self.nv] \
                                    + x[i+1, j  ] * x1[i+1, self.nv] \
                                    + x[i+1, j  ] * x1[i,   self.nv] \
                                    + x[i-1, j  ] * x1[i-1, self.nv] \
                                    + x[i-1, j  ] * x1[i,   self.nv] \
                                    - x[i+1, j  ] * x1[i+1, self.nv] \
                                    - x[i+1, j  ] * x1[i,   self.nv] \
                                    - x[i-1, j+1] * x1[i-1, self.nv] \
                                    + x[i+1, j+1] * x1[i+1, self.nv] \
                                    - x[i,   j+1] * x1[i-1, self.nv] \
                                    + x[i,   j+1] * x1[i+1, self.nv] \
                                    + x[i-1, j+1] * x1[i,   self.nv] \
                                    - x[i+1, j+1] * x1[i,   self.nv] \
                                    - x[i,   j+1] * x1[i-1, self.nv] \
                                    + x[i,   j+1] * x1[i+1, self.nv] \
                                  ) / (2 * 12 * self.hx * self.hv)
                    
                    arakawa_1_0 = ( \
                                    + x1[i-1, j-1] * x[i-1, self.nv] \
                                    - x1[i+1, j-1] * x[i+1, self.nv] \
                                    + x1[i,   j-1] * x[i-1, self.nv] \
                                    - x1[i,   j-1] * x[i+1, self.nv] \
                                    - x1[i-1, j-1] * x[i,   self.nv] \
                                    + x1[i+1, j-1] * x[i,   self.nv] \
                                    + x1[i,   j-1] * x[i-1, self.nv] \
                                    - x1[i,   j-1] * x[i+1, self.nv] \
                                    - x1[i-1, j  ] * x[i-1, self.nv] \
                                    - x1[i-1, j  ] * x[i,   self.nv] \
                                    + x1[i+1, j  ] * x[i+1, self.nv] \
                                    + x1[i+1, j  ] * x[i,   self.nv] \
                                    + x1[i-1, j  ] * x[i-1, self.nv] \
                                    + x1[i-1, j  ] * x[i,   self.nv] \
                                    - x1[i+1, j  ] * x[i+1, self.nv] \
                                    - x1[i+1, j  ] * x[i,   self.nv] \
                                    - x1[i-1, j+1] * x[i-1, self.nv] \
                                    + x1[i+1, j+1] * x[i+1, self.nv] \
                                    - x1[i,   j+1] * x[i-1, self.nv] \
                                    + x1[i,   j+1] * x[i+1, self.nv] \
                                    + x1[i-1, j+1] * x[i,   self.nv] \
                                    - x1[i+1, j+1] * x[i,   self.nv] \
                                    - x1[i,   j+1] * x[i-1, self.nv] \
                                    + x1[i,   j+1] * x[i+1, self.nv] \
                                  ) / (2 * 12 * self.hx * self.hv)
                    
                    
                    y[i, j] = time_deriv - arakawa_0_h0 - arakawa_0_1 - arakawa_1_0
        
    
    def formRHS(self, B):
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        self.da.globalToLocal(self.X1, self.localX1)
        
        b  = self.da.getVecArray(B)
        x1 = self.da.getVecArray(self.localX1)
        h0 = self.h0
        
        
        for j in np.arange(ys, ye):
            for i in np.arange(xs, xe):
                
                if j == self.nv:
                    # Poisson equation
                    b[i, j] = - self.poisson_const
                    
                elif j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[i, j] = 0.0
                    
                else:
                    # Vlasov equation
                    
                    time_deriv = ( \
                                   + 1. * x1[i-1, j-1] \
                                   + 2. * x1[i-1, j  ] \
                                   + 1. * x1[i-1, j+1] \
                                   + 2. * x1[i,   j-1] \
                                   + 4. * x1[i,   j  ] \
                                   + 2. * x1[i,   j+1] \
                                   + 1. * x1[i+1, j-1] \
                                   + 2. * x1[i+1, j  ] \
                                   + 1. * x1[i+1, j+1] \
                                  ) / (16. * self.ht)
                    
                    arakawa_1_h0 = ( \
                                     + x1[i-1, j-1] * h0[j  ] \
                                     - x1[i-1, j-1] * h0[j-1] \
                                     - x1[i-1, j+1] * h0[j  ] \
                                     + x1[i-1, j+1] * h0[j+1] \
                                     - x1[i-1, j  ] * h0[j-1] \
                                     + x1[i-1, j  ] * h0[j+1] \
                                     - x1[i-1, j  ] * h0[j-1] \
                                     + x1[i-1, j  ] * h0[j+1] \
                                     - x1[i+1, j-1] * h0[j  ] \
                                     + x1[i+1, j-1] * h0[j-1] \
                                     + x1[i+1, j+1] * h0[j  ] \
                                     - x1[i+1, j+1] * h0[j+1] \
                                     + x1[i+1, j  ] * h0[j-1] \
                                     - x1[i+1, j  ] * h0[j+1] \
                                     + x1[i+1, j  ] * h0[j-1] \
                                     - x1[i+1, j  ] * h0[j+1] \
                                     + x1[i,   j-1] * h0[j-1] \
                                     + x1[i,   j-1] * h0[j  ] \
                                     - x1[i,   j-1] * h0[j-1] \
                                     - x1[i,   j-1] * h0[j  ] \
                                     - x1[i,   j+1] * h0[j+1] \
                                     - x1[i,   j+1] * h0[j  ] \
                                     + x1[i,   j+1] * h0[j+1] \
                                     + x1[i,   j+1] * h0[j  ] \
                                   ) / (2. * 12. * self.hx * self.hv)
                    
                    b[i, j] = time_deriv + arakawa_1_h0
    
