'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from numpy.fft import ifftshift


from petsc4py import PETSc


cdef class TensorProductPreconditioner(object):
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
        
        # distributed arrays and grid
        self.da1  = da1
        self.grid = grid
        
        # distributed arrays
        self.dax = PETSc.DMDA().create(dim=2, dof=1,
                                 sizes=[self.grid.nx, self.grid.nv],
                                 proc_sizes=[1, PETSc.COMM_WORLD.getSize()],
                                 boundary_type=['ghosted', 'periodic'],
                                 stencil_width=2,
                                 stencil_type='box')
        
        self.day = PETSc.DMDA().create(dim=2, dof=1,
                                 sizes=[self.grid.nx, self.grid.nv],
                                 proc_sizes=[PETSc.COMM_WORLD.getSize(), 1],
                                 boundary_type=['ghosted', 'periodic'],
                                 stencil_width=2,
                                 stencil_type='box')
        
        
        # interim vectors
        self.B = self.da1.createGlobalVec()
        self.X = self.da1.createGlobalVec()
        
        
    def __dealloc__(self):
        self.B.destroy()
        self.X.destroy()
 
        self.dax.destroy()
        self.day.destroy()
        
    
    @staticmethod
    def create(str    type  not None,
               object da1   not None,
               object daphi not None,
               Grid   grid  not None,
               Vec    phi   not None):
        
        if type == 'tensor_kinetic_scipy':
            preconditioner_object = __import__("vlasov.solvers.preconditioner.TensorProductKineticSciPy", globals(), locals(), ['TensorProductPreconditionerKineticSciPy'],  0)
            return preconditioner_object.TensorProductPreconditionerKineticSciPy(da1, grid)
        elif type == 'tensor_kinetic':
            preconditioner_object = __import__("vlasov.solvers.preconditioner.TensorProductKineticFast",  globals(), locals(), ['TensorProductPreconditionerKineticFast'],  0)
            return preconditioner_object.TensorProductPreconditionerKineticFast(da1, grid)
        elif type == 'tensor_potential_scipy':
            preconditioner_object = __import__("vlasov.solvers.preconditioner.TensorProductPotentialSciPy",  globals(), locals(), ['TensorProductPreconditionerPotentialSciPy'],  0)
            return preconditioner_object.TensorProductPreconditionerPotentialSciPy(da1, daphi, grid, phi)
        elif type == 'tensor_potential':
            preconditioner_object = __import__("vlasov.solvers.preconditioner.TensorProductPotentialFast",  globals(), locals(), ['TensorProductPreconditionerPotentialFast'],  0)
            return preconditioner_object.TensorProductPreconditionerPotentialFast(da1, daphi, grid, phi)
        else:
            return None
        
    
#     cdef tensorProductDiagonal(self, Vec X, Vec Y):
#         
#         self.copy_da1_to_day(X, self.F)                  # copy X(da1) to F(day)
#          
#         self.fft_x(self.F, self.Ffft)                    # FFT F for each x
#          
#         self.copy_cay_to_cax(self.Ffft, self.Bfft)       # copy F'(day) to B'(dax)
#          
#         self.fft_y(self.Bfft, self.Dfft)                 # FFT F for each y
#         
#         self.solve(self.Dfft)                            # solve AC'=B' for each y
#          
#         self.ifft_y(self.Efft, self.Bfft)                # iFFT Z' for each y
#         
#         self.copy_cax_to_cay(self.Bfft, self.Zfft)       # copy C'(dax) to Z'(day)
#          
#         self.ifft_x(self.Zfft, self.Z)                   # iFFT Z' for each x
#          
#         self.copy_day_to_da1(self.Z, Y)                  # copy Z(day) to Y(da1)
        

    def compute_eigenvalues(self, n):
        eigen = np.empty(n, dtype=np.complex128)
        
        for i in range(n):
            eigen[i] = np.exp(2.j * np.pi * float(i) / n * (n-1)) \
                     - np.exp(2.j * np.pi * float(i) / n)
        
        return ifftshift(eigen)

    
    cdef copy_da1_to_dax(self, Vec X, Vec Y):
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        (xsx, xex), (ysx, yex) = self.dax.getRanges()
        
        cdef double[:] x
        cdef double[:] y
        
        if xs1 == xsx and xe1 == xex and ys1 == ysx and ye1 == yex:
            x = X.getArray()
            y = Y.getArray()
            y[...] = x[...]
            
        else:
            self.d1xScatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
         
        
    cdef copy_dax_to_da1(self, Vec X, Vec Y):
        (xsx, xex), (ysx, yex) = self.dax.getRanges()
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        
        cdef double[:] x
        cdef double[:] y
        
        if xs1 == xsx and xe1 == xex and ys1 == ysx and ye1 == yex:
            x = X.getArray()
            y = Y.getArray()
            y[...] = x[...]
        
        else:
            self.dx1Scatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        
        
    cdef copy_da1_to_day(self, Vec X, Vec Y):
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        (xsy, xey), (ysy, yey) = self.day.getRanges()
        
        cdef double[:] x
        cdef double[:] y
        
        if xs1 == xsy and xe1 == xey and ys1 == ysy and ye1 == yey:
            x = X.getArray()
            y = Y.getArray()
            y[...] = x[...]
            
        else:
            self.d1yScatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
         
        
    cdef copy_day_to_da1(self, Vec X, Vec Y):
        (xsy, xey), (ysy, yey) = self.day.getRanges()
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        
        cdef double[:] x
        cdef double[:] y
        
        if xs1 == xsy and xe1 == xey and ys1 == ysy and ye1 == yey:
            x = X.getArray()
            y = Y.getArray()
            y[...] = x[...]
        
        else:
            self.dy1Scatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        
        
    cdef copy_cax_to_cay(self, Vec X, Vec Y):
        self.cxyScatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        
    
    cdef copy_cay_to_cax(self, Vec X, Vec Y):
        self.cyxScatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        

    cdef update_matrices(self):
        print("ERROR: function not implemented.")
    
    cdef tensorProduct(self, Vec X, Vec Y):
        print("ERROR: function not implemented.")
    
    cdef solve(self, Vec X):
        print("ERROR: function not implemented.")

