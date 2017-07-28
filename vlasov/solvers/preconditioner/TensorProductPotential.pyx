'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

import pyfftw
# from pyfftw.interfaces.scipy_fftpack import fft, ifft

from petsc4py import PETSc


cdef class TensorProductPreconditionerPotential(TensorProductPreconditioner):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 object da1   not None,
                 object daphi not None,
                 Grid   grid  not None,
                 Vec    phi   not None):
        '''
        Constructor
        '''
        
        super().__init__(da1, grid)
        
        self.daphi = daphi
        self.phi   = phi
        
        self.cax = PETSc.DMDA().create(dim=2, dof=2,
                                 sizes=[self.grid.nx, self.grid.nv//2+1],
                                 proc_sizes=[1, PETSc.COMM_WORLD.getSize()],
                                 boundary_type=['ghosted', 'periodic'],
                                 stencil_width=2,
                                 stencil_type='box')
        
        self.cay = PETSc.DMDA().create(dim=2, dof=2,
                                 sizes=[self.grid.nx, self.grid.nv//2+1],
                                 proc_sizes=[PETSc.COMM_WORLD.getSize(), 1],
                                 boundary_type=['ghosted', 'periodic'],
                                 stencil_width=2,
                                 stencil_type='box')
        
        self.F = self.day.createGlobalVec()
        self.Z = self.day.createGlobalVec()
        
        self.Ffft = self.cay.createGlobalVec()
        self.Bfft = self.cax.createGlobalVec()
        self.Cfft = self.cax.createGlobalVec()
        self.Zfft = self.cay.createGlobalVec()

        # temporary variables for scatter objects
        cdef int i, j, k, l
        cdef int xs1, xe1, ys1, ye1
#         cdef int xsx, xex, ysx, yex
        cdef int xsy, xey, ysy, yey
        
        # create 1x-x1 and 1y-y1 scatter objects
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        (xsy, xey), (ysy, yey) = self.day.getRanges()
        
        # IS().createStride(PetscInt n, PetscInt first, PetscInt step)
        #     n      - the length of the locally owned portion of the index set
        #     first  - the first element of the locally owned portion of the index set
        #     step   - the change to the next index        
        # self.d1Indices = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
        # self.dxIndices = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)

#         self.d1Indices = PETSc.IS().createStride((xey-xsy)*self.grid.nv, xsy*self.grid.nx, self.grid.nv)
#         self.dyIndices = PETSc.IS().createStride((xey-xsy)*self.grid.nv, xsy*self.grid.nx, self.grid.nv)
        
        self.d1Indices = PETSc.IS().createStride((xey-xsy)*self.grid.nv, xsy*self.grid.nv, 1)
        self.dyIndices = PETSc.IS().createStride((xey-xsy)*self.grid.nv, xsy*self.grid.nv, 1)
        
        self.da1.getAO().app2petsc(self.d1Indices)
        self.day.getAO().app2petsc(self.dyIndices)
        
        self.d1yScatter = PETSc.Scatter().create(self.X, self.d1Indices, self.F, self.dyIndices)
        self.dy1Scatter = PETSc.Scatter().create(self.Z, self.dyIndices, self.B, self.d1Indices)

        # create xy-yx scatter objects
        (xsy, xey), (ysy, yey) = self.cay.getRanges()
#         (xsx, xex), (ysx, yex) = self.cax.getRanges()
         
        cdef int nx = self.grid.nx
        cdef int nv = self.grid.nv//2+1
         
        assert ysy == 0
        assert yey == nv
         
        cdef int nindices = (xey-xsy)*nv*2
         
        cdef int[:] xindexlist = np.empty(nindices, dtype=np.int32)
        cdef int[:] yindexlist = np.empty(nindices, dtype=np.int32)
         
        l = 0
        for i in range(xsy, xey):
            for j in range(ysy, yey):
                for k in range(0,2):
                    xindexlist[l] = 2*(j*nx + i) + k
                    yindexlist[l] = 2*(i*nv + j) + k
                    l += 1
        
        self.cyIndices  = PETSc.IS().createGeneral(yindexlist)
        self.cxIndices  = PETSc.IS().createGeneral(xindexlist)
         
        self.cay.getAO().app2petsc(self.cyIndices)
        self.cax.getAO().app2petsc(self.cyIndices)
     
        self.cyxScatter = PETSc.Scatter().create(self.Ffft, self.cyIndices, self.Bfft, self.cxIndices)        
        self.cxyScatter = PETSc.Scatter().create(self.Cfft, self.cxIndices, self.Zfft, self.cyIndices)
        
        # compute eigenvalues
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Computing eigenvalues.")
        
        self.eigen = self.compute_eigenvalues(self.grid.nv)
    
    
    def __dealloc__(self):
        self.F.destroy()
        self.Z.destroy()
        
        self.Ffft.destroy()
        self.Bfft.destroy()
        self.Cfft.destroy()
        self.Zfft.destroy()
        
        self.d1yScatter.destroy()
        self.dy1Scatter.destroy()
        
        self.d1Indices.destroy()
        self.dyIndices.destroy()
        
        self.cyxScatter.destroy()        
        self.cxyScatter.destroy()
        
        self.cxIndices.destroy()
        self.cyIndices.destroy()
        
        self.cax.destroy()
        self.cay.destroy()
        
#         del self.fftw_plan
#         del self.ifftw_plan
        
    
    cdef tensorProduct(self, Vec X, Vec Y):
        
        self.copy_da1_to_day(X, self.F)                  # copy X(da1) to F(day)
        
        self.fft(self.F, self.Ffft)                      # FFT F(day) to F'(cay) for each x
        
        self.copy_cay_to_cax(self.Ffft, self.Bfft)       # copy F'(cay) to B'(cax)
         
        self.solve(self.Bfft)                            # solve AC'=B' for each v, setting B'=C'
         
        self.copy_cax_to_cay(self.Bfft, self.Zfft)       # copy B'(cax) to Z'(cay)
        
#         x = self.Ffft.getArray()
#         z = self.Zfft.getArray()
#         z[...] = x[...]
         
        self.ifft(self.Zfft, self.Z)                     # iFFT Z'(cay) to Z(day) for each x

#         x = self.F.getArray()
#         z = self.Z.getArray()
#         z[...] = x[...]
        
        self.copy_day_to_da1(self.Z, Y)                  # copy Z(day) to Y(da1)
    
#         x = X.getArray()
#         y = Y.getArray()
#         y[...] = x[...]


    cdef fft(self, Vec X, Vec Y):
        print("ERROR: function not implemented.")

    cdef ifft(self, Vec X, Vec Y):
        print("ERROR: function not implemented.")

