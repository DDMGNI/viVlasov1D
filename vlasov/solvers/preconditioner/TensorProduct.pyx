'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc


cdef class TensorProductPreconditioner(object):
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
        
        # distributed arrays and grid
        self.da1  = da1
        self.grid = grid
        
        # distributed arrays
        self.dax = VIDA().create(dim=2, dof=1,
                                 sizes=[self.grid.nx, self.grid.nv],
                                 proc_sizes=[1, PETSc.COMM_WORLD.getSize()],
                                 boundary_type=['periodic', 'ghosted'],
                                 stencil_width=2,
                                 stencil_type='box')
        
        self.day = VIDA().create(dim=2, dof=1,
                                 sizes=[self.grid.nx, self.grid.nv],
                                 proc_sizes=[PETSc.COMM_WORLD.getSize(), 1],
                                 boundary_type=['periodic', 'ghosted'],
                                 stencil_width=2,
                                 stencil_type='box')
        
        self.cax = VIDA().create(dim=2, dof=2,
                                 sizes=[self.grid.nx//2+1, self.grid.nv],
                                 proc_sizes=[1, PETSc.COMM_WORLD.getSize()],
                                 boundary_type=['periodic', 'ghosted'],
                                 stencil_width=2,
                                 stencil_type='box')
        
        self.cay = VIDA().create(dim=2, dof=2,
                                 sizes=[self.grid.nv, self.grid.nx//2+1],
                                 proc_sizes=[1, PETSc.COMM_WORLD.getSize()],
                                 boundary_type=['periodic', 'ghosted'],
                                 stencil_width=2,
                                 stencil_type='box')
        
        
        # interim vectors
        self.B    = self.da1.createGlobalVec()
        self.X    = self.da1.createGlobalVec()
        self.F    = self.dax.createGlobalVec()
        self.Z    = self.dax.createGlobalVec()
        
        self.Ffft = self.cax.createGlobalVec()
        self.Bfft = self.cay.createGlobalVec()
        self.Zfft = self.cax.createGlobalVec()
        
        
        # temporary variables for scatter objects
        cdef np.uint64_t i, j, k, l
        cdef np.uint64_t xs1, xe1, ys1, ye1
        cdef np.uint64_t xsx, xex, ysx, yex
        cdef np.uint64_t ysy, yey, xsy, xey 
        
        # create 1x-x1 scatter objects
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        (xsx, xex), (ysx, yex) = self.dax.getRanges()
        
        self.d1Indices = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
        self.dxIndices = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
        
        self.da1.getAO().app2petsc(self.dxIndices)
        self.dax.getAO().app2petsc(self.d1Indices)
        
        self.d1xScatter = PETSc.Scatter().create(self.X, self.d1Indices, self.F, self.dxIndices)
        self.dx1Scatter = PETSc.Scatter().create(self.Z, self.dxIndices, self.B, self.d1Indices)
        
        # create xy-yx scatter objects
        (xsx, xex), (ysx, yex) = self.cax.getRanges()
        (ysy, yey), (xsy, xey) = self.cay.getRanges()
        
        cdef np.uint64_t nx = self.grid.nx//2+1
        cdef np.uint64_t nv = self.grid.nv
        
        assert xsx == 0
        assert xex == nx
        
        nindices = (yex-ysx)*nx*2
        
        cdef int[:] xindexlist = np.empty(nindices, dtype=np.int32)
        cdef int[:] yindexlist = np.empty(nindices, dtype=np.int32)
        
        l = 0
        for i in range(xsx, xex):
            for j in range(ysx, yex):
                for k in range(0,2):
                    xindexlist[l] = 2*(j*nx + i) + k
                    yindexlist[l] = 2*(i*nv + j) + k
                    l += 1
        
        self.cxIndices  = PETSc.IS().createGeneral(xindexlist)
        self.cyIndices  = PETSc.IS().createGeneral(yindexlist)
        
        self.cax.getAO().app2petsc(self.cxIndices)
        self.cay.getAO().app2petsc(self.cyIndices)
    
        self.cxyScatter = PETSc.Scatter().create(self.Ffft, self.cxIndices, self.Bfft, self.cyIndices)
        self.cyxScatter = PETSc.Scatter().create(self.Bfft, self.cyIndices, self.Zfft, self.cxIndices)        
        
        
    def __dealloc__(self):
        self.d1xScatter.destroy()
        self.dx1Scatter.destroy()
        self.cxyScatter.destroy()
        self.cyxScatter.destroy()
        
        self.d1Indices.destroy()
        self.dxIndices.destroy()
        self.cxIndices.destroy()
        self.cyIndices.destroy()
        
        self.B.destroy()
        self.X.destroy()
        self.F.destroy()
        self.Z.destroy()
        
        self.Ffft.destroy()
        self.Bfft.destroy()
        self.Zfft.destroy()
        
        self.dax.destroy()
        self.day.destroy()
        self.cax.destroy()
        self.cay.destroy()
        
    
    @staticmethod
    def create(str  type not None,
               VIDA da1  not None,
               Grid grid not None):
        
        if type == 'tensorfast':
            preconditioner_object = __import__("vlasov.solvers.preconditioner.TensorProductFast",  globals(), locals(), ['TensorProductPreconditionerFast'],  0)
            return preconditioner_object.TensorProductPreconditionerFast(da1, grid)
        elif type == 'tensorscipy':
            preconditioner_object = __import__("vlasov.solvers.preconditioner.TensorProductSciPy", globals(), locals(), ['TensorProductPreconditionerSciPy'],  0)
            return preconditioner_object.TensorProductPreconditionerSciPy(da1, grid)
        else:
            return None
        
    
    cdef tensorProduct(self, Vec X, Vec Y):
        
        self.copy_da1_to_dax(X, self.F)                  # copy X(da1) to F(dax)
        
        self.fft(self.F, self.Ffft)                      # FFT F for each v
        
        self.copy_cax_to_cay(self.Ffft, self.Bfft)       # copy F'(dax) to B'(day)
        
        self.solve(self.Bfft)                            # solve AC'=B' for each x where C' is saved in B'
        
        self.copy_cay_to_cax(self.Bfft, self.Zfft)       # copy B'(day) to Z'(dax)
        
        self.ifft(self.Zfft, self.Z)                     # iFFT Z' for each v
        
        self.copy_dax_to_da1(self.Z, Y)                  # copy Z(dax) to Y(da1)
        
    
    cdef copy_da1_to_dax(self, Vec X, Vec Y):
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        (xsx, xex), (ysx, yex) = self.dax.getRanges()
        
        cdef np.ndarray[double, ndim=1] x
        cdef np.ndarray[double, ndim=1] y
        
        if xs1 == xsx and xe1 == xex and ys1 == ysx and ye1 == yex:
            x = X.getArray()
            y = Y.getArray()
            y[...] = x[...]
            
        else:
            self.d1xScatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
         
        
        
    cdef copy_dax_to_da1(self, Vec X, Vec Y):
        (xsx, xex), (ysx, yex) = self.dax.getRanges()
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        
        cdef np.ndarray[double, ndim=1] x
        cdef np.ndarray[double, ndim=1] y
        
        if xs1 == xsx and xe1 == xex and ys1 == ysx and ye1 == yex:
            x = X.getArray()
            y = Y.getArray()
            y[...] = x[...]
        
        else:
            self.dx1Scatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        
        
    cdef copy_cax_to_cay(self, Vec X, Vec Y):
        self.cxyScatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        
    
    cdef copy_cay_to_cax(self, Vec X, Vec Y):
        self.cyxScatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        

    cdef fft (self, Vec X, Vec Y):
        print("ERROR: function not implemented.")

    cdef ifft(self, Vec X, Vec Y):
        print("ERROR: function not implemented.")

    cdef solve(self, Vec X):
        print("ERROR: function not implemented.")

