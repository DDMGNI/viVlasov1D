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
    
    
    def __init__(self, da,
                 nx, nv, ht, hx, hv,
                 h0, poisson_const):
        '''
        Constructor
        '''
        
        assert da.getDim() == 2
        
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
        
        # create history vectors
        self.X1 = self.da.createGlobalVec()
        
        # create local vectors
        self.localB  = da.createLocalVec()
        self.localX  = da.createLocalVec()
        self.localX1 = da.createLocalVec()
        
        # compute constants 
#        self.time_deriv = 1. / (16. * self.ht)
#        self.arakawa    = 1. / (2. * 12. * self.hx * self.hv)
        self.time_deriv = 1. / (2. * self.ht)
        self.arakawa    = 1. / (3. * self.hx * self.hv)
        
    
    def update_history(self, X):
        
        x  = self.da.getVecArray(X)
        x1 = self.da.getVecArray(self.X1)
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        x1[xs:xe, ys:ye] = x[xs:xe, ys:ye]
        
#        self.X1.assemblyBegin()
#        self.X1.assemblyEnd()
        
    
    def formMat(self, A, X):
        
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()

        (xs, xe), (ys, ye) = self.da.getRanges()
        
        self.da.globalToLocal(X, self.localX1)
#        self.da.globalToLocal(self.X1, self.localX1)
        
        x1 = self.da.getVecArray(self.localX1)
        h0 = self.h0
        
        
        # constants for matrix
        poisson_laplace  = - 1. / self.hx**2
        poisson_integral = + 0.25 * self.poisson_const * self.hv
        
        time_deriv = + self.time_deriv
        arakawa    = - self.arakawa
        
        
        for i in np.arange(ys, ye):
            for j in np.arange(xs, xe):
                data = []
                
                if j == self.nv:
                    # Poisson equation
                    
#                    if i == self.nx-1:
#                        for k in range(0, self.nx):
#                            data.append( ((k, self.nv), 1.) )
#                        
#                    else:
                    data.append( ((i-1, self.nv), + 1. * poisson_laplace) )
                    data.append( ((i,   self.nv), - 2. * poisson_laplace) )
                    data.append( ((i+1, self.nv), + 1. * poisson_laplace) )
                    
                    for k in range(0, self.nv):
                        data.append( ((i-1, k), 1. * poisson_integral) )
                        data.append( ((i,   k), 2. * poisson_integral) )
                        data.append( ((i+1, k), 1. * poisson_integral) )
                    
#                    data.append( ((i, j), 1.) )
                    
                    pass
                elif j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    data.append( ((i, j), 1.) )
                    
                    pass
                else:
                    # Vlasov equation
                    
                    data.append( ((i-1, j-1), 1. * time_deriv \
                                            + 1. * arakawa * ( h0[j  ] - h0[j-1] ) \
#                                            + 1. * arakawa * ( x1[self.nv, i-1] - x1[self.nv, i  ] ) \
                               ) )
                    
                    data.append( ((i-1, j  ), 2. * time_deriv \
                                            + 2. * arakawa * ( h0[j+1] - h0[j-1] ) \
                               ) )
                    
                    data.append( ((i-1, j+1), 1. * time_deriv \
                                            + 1. * arakawa * ( h0[j+1] - h0[j  ] ) \
#                                            + 1. * arakawa * ( x1[self.nv, i  ] - x1[self.nv, i-1] ) \
                               ) )
                    
                    data.append( ((i,   j-1), 2. * time_deriv \
#                                            + 2. * arakawa * ( x1[self.nv, i-1] - x1[self.nv, i+1] ) \
                               ) )
                    
                    data.append( ((i,   j  ), 4. * time_deriv) )
                    
                    data.append( ((i,   j+1), 2. * time_deriv \
#                                            + 2. * arakawa * ( x1[self.nv, i+1] - x1[self.nv, i-1] ) \
                               ) )
                    
                    data.append( ((i+1, j-1), 1. * time_deriv \
                                            + 1. * arakawa * ( h0[j-1] - h0[j  ] ) \
#                                            + 1. * arakawa * ( x1[self.nv, i  ] - x1[self.nv, i+1] ) \
                               ) )
                    
                    data.append( ((i+1, j  ), 2. * time_deriv \
                                            + 2. * arakawa * ( h0[j-1] - h0[j+1] ) \
                               ) )
                    
                    data.append( ((i+1, j+1), 1. * time_deriv \
                                            + 1. * arakawa * ( h0[j  ] - h0[j+1] ) \
#                                            + 1. * arakawa * ( x1[self.nv, i+1] - x1[self.nv, i  ] ) \
                               ) )
                    
                    
#                    data.append( ((i-1, self.nv), arakawa * ( 2. * x1[j-1, i  ] - 2. * x1[j+1, i  ] \
#                                                            + 1. * x1[j-1, i-1] - 1. * x1[j+1, i-1] ) \
#                               ) )
#                    
#                    data.append( ((i,   self.nv), arakawa * ( 1. * x1[j-1, i+1] - 1. * x1[j-1, i-1] \
#                                                            + 1. * x1[j+1, i-1] - 1. * x1[j+1, i+1] ) \
#                               ) )
#                    
#                    data.append( ((i+1, self.nv), arakawa * ( 2. * x1[j+1, i  ] - 2. * x1[j-1, i  ] \
#                                                            + 1. * x1[j+1, i+1] - 1. * x1[j-1, i+1] ) \
#                               ) )
                    
                    pass

                if len(data) > 0:
                    row.index = (j, i)
                    for index, value in data:
                        col.index = (index[1], index[0])
                        A.setValueStencil(row, col, value)
                
            
        A.assemblyBegin()
        A.assemblyEnd()
        
    
    def formRHSMat(self, A):
        
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()

        (xs, xe), (ys, ye) = self.da.getRanges()
        
        self.da.globalToLocal(self.X1, self.localX1)
        
        x1 = self.da.getVecArray(self.localX1)
        h0 = self.h0
        
        
        time_deriv = + self.time_deriv
        arakawa    = + self.arakawa
        
        
        for i in np.arange(ys, ye):
            for j in np.arange(xs, xe):
                data = []
                
                if j == self.nv:
                    # Poisson equation
                    data.append( ((i, j), 0.) )
                    
                elif j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    data.append( ((i, j), 0.) )
                    
                else:
                    # Vlasov equation
                    
                    data.append( ((i-1, j-1), 1. * time_deriv \
                                            + 1. * arakawa * ( h0[j  ] - h0[j-1] ) \
                               ) )
                    
                    data.append( ((i-1, j  ), 2. * time_deriv \
                                            + 2. * arakawa * ( h0[j+1] - h0[j-1] ) \
                               ) )
                    
                    data.append( ((i-1, j+1), 1. * time_deriv \
                                            + 1. * arakawa * ( h0[j+1] - h0[j  ] ) \
                               ) )
                    
                    data.append( ((i,   j-1), 2. * time_deriv \
                               ) )
                    
                    data.append( ((i,   j  ), 4. * time_deriv) )
                    
                    data.append( ((i,   j+1), 2. * time_deriv \
                               ) )
                    
                    data.append( ((i+1, j-1), 1. * time_deriv \
                                            + 1. * arakawa * ( h0[j-1] - h0[j  ] ) \
                               ) )
                    
                    data.append( ((i+1, j  ), 2. * time_deriv \
                                            + 2. * arakawa * ( h0[j-1] - h0[j+1] ) \
                               ) )
                    
                    data.append( ((i+1, j+1), 1. * time_deriv \
                                            + 1. * arakawa * ( h0[j  ] - h0[j+1] ) \
                               ) )
                    
                
                if len(data) > 0:
                    row.index = (j, i)
                    for index, value in data:
                        col.index = (index[1], index[0])
                        A.setValueStencil(row, col, value)
                
            
        A.assemblyBegin()
        A.assemblyEnd()
        
    
    def formRHS(self, B, X):
        
        (xs, xe), (ys, ye) = self.da.getRanges()
        
        self.da.globalToLocal(X, self.localX1)
#        self.da.globalToLocal(self.X1, self.localX1)
        
        b  = self.da.getVecArray(B)
        x1 = self.da.getVecArray(self.localX1)
        h0 = self.h0
        
        n0 = 1.
        
#        n0 = np.sum(x1[0:self.nv, 0:self.nx], axis=0).mean() * self.hv
        
#        print(n0)
        
        
        for i in np.arange(ys, ye):
            for j in np.arange(xs, xe):
                value = 0.0
                
                if j == self.nv:
                    # Poisson equation
#                    if i == self.nx-1:
#                        value = 0.0
#                    else:
                    value = + n0 * self.poisson_const
                    
                elif j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    value = 0.0
                    
                else:
                    # Vlasov equation
                    
                    time_deriv = ( \
                                   + 1. * x1[j-1, i-1] \
                                   + 2. * x1[j  , i-1] \
                                   + 1. * x1[j+1, i-1] \
                                   + 2. * x1[j-1, i  ] \
                                   + 4. * x1[j  , i  ] \
                                   + 2. * x1[j+1, i  ] \
                                   + 1. * x1[j-1, i+1] \
                                   + 2. * x1[j  , i+1] \
                                   + 1. * x1[j+1, i+1] \
                                  ) * self.time_deriv
                    
                    arakawa_1_h0 = ( \
                                     + 1. * x1[j-1, i-1] * (h0[j  ] - h0[j-1]) \
                                     + 1. * x1[j-1, i+1] * (h0[j-1] - h0[j  ]) \
                                     + 2. * x1[j  , i-1] * (h0[j+1] - h0[j-1]) \
                                     + 2. * x1[j  , i+1] * (h0[j-1] - h0[j+1]) \
                                     + 1. * x1[j+1, i-1] * (h0[j+1] - h0[j  ]) \
                                     + 1. * x1[j+1, i+1] * (h0[j  ] - h0[j+1]) \
                                   ) * self.arakawa
                    
                    
                    value = time_deriv + arakawa_1_h0
                
                
                b[j, i] = value
#                B.setValue(i*(self.nv+1)+j, value)
                
        
#        B.assemblyBegin()
#        B.assemblyEnd()


    def isSparse(self):
        return True
    
