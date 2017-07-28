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


cdef class TensorProductPreconditionerKinetic(TensorProductPreconditioner):
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
        
        self.cax = PETSc.DMDA().create(dim=2, dof=2,
                                 sizes=[self.grid.nx//2+1, self.grid.nv],
                                 proc_sizes=[1, PETSc.COMM_WORLD.getSize()],
                                 boundary_type=['ghosted', 'periodic'],
                                 stencil_width=2,
                                 stencil_type='box')
        
        self.cay = PETSc.DMDA().create(dim=2, dof=2,
                                 sizes=[self.grid.nv, self.grid.nx//2+1],
                                 proc_sizes=[1, PETSc.COMM_WORLD.getSize()],
                                 boundary_type=['periodic', 'ghosted'],
                                 stencil_width=2,
                                 stencil_type='box')
        
        self.F = self.dax.createGlobalVec()
        self.Z = self.dax.createGlobalVec()
        
        self.Ffft = self.cax.createGlobalVec()
        self.Bfft = self.cay.createGlobalVec()
        self.Cfft = self.cay.createGlobalVec()
        self.Zfft = self.cax.createGlobalVec()
        
        # temporary variables for scatter objects
        cdef int i, j, k, l
        cdef int xs1, xe1, ys1, ye1
        cdef int xsx, xex, ysx, yex
#         cdef int xsy, xey, ysy, yey
        
        # create 1x-x1 and 1y-y1 scatter objects
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        (xsx, xex), (ysx, yex) = self.dax.getRanges()
        
        # IS().createStride(PetscInt n, PetscInt first, PetscInt step)
        #     n      - the length of the locally owned portion of the index set
        #     first  - the first element of the locally owned portion of the index set
        #     step   - the change to the next index        
        self.d1Indices = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
        self.dxIndices = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
        
        self.da1.getAO().app2petsc(self.d1Indices)
        self.dax.getAO().app2petsc(self.dxIndices)
        
        self.d1xScatter = PETSc.Scatter().create(self.X, self.d1Indices, self.F, self.dxIndices)
        self.dx1Scatter = PETSc.Scatter().create(self.Z, self.dxIndices, self.B, self.d1Indices)
        
        # create xy-yx scatter objects
        (xsx, xex), (ysx, yex) = self.cax.getRanges()
#         (ysy, yey), (xsy, xey) = self.cay.getRanges()
         
        cdef int nx = self.grid.nx//2+1
        cdef int nv = self.grid.nv
         
        assert xsx == 0
        assert xex == nx
         
        cdef int nindices = (yex-ysx)*nx*2
         
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
        self.cyxScatter = PETSc.Scatter().create(self.Cfft, self.cyIndices, self.Zfft, self.cxIndices)        
        
        # compute eigenvalues
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Computing eigenvalues.")
        
        self.eigen = self.compute_eigenvalues(self.grid.nx)
        
        
    def __dealloc__(self):
        self.F.destroy()
        self.Z.destroy()
        
        self.Ffft.destroy()
        self.Bfft.destroy()
        self.Cfft.destroy()
        self.Zfft.destroy()
        
        self.d1xScatter.destroy()
        self.dx1Scatter.destroy()
        
        self.d1Indices.destroy()
        self.dxIndices.destroy()
        
        self.cyxScatter.destroy()        
        self.cxyScatter.destroy()
        
        self.cxIndices.destroy()
        self.cyIndices.destroy()
        
        self.cax.destroy()
        self.cay.destroy()
        
#         del self.fftw_plan
#         del self.ifftw_plan

    
    cdef tensorProduct(self, Vec X, Vec Y):
        
        self.copy_da1_to_dax(X, self.F)                  # copy X(da1) to F(dax)
        
        self.fft(self.F, self.Ffft)                      # FFT F to F' for each v
        
        self.copy_cax_to_cay(self.Ffft, self.Bfft)       # copy F'(cax) to B'(cay)
        
        self.solve(self.Bfft)                            # solve AC'=B' for each x and set B'=C'
        
        self.copy_cay_to_cax(self.Bfft, self.Zfft)       # copy C'(cay) to Z'(cax)
        
        self.ifft(self.Zfft, self.Z)                     # iFFT Z' to Z for each v
        
        self.copy_dax_to_da1(self.Z, Y)                  # copy Z(dax) to Y(da1)
        
    
    cdef fft(self, Vec X, Vec Y):
        print("ERROR: function not implemented.")

    cdef ifft(self, Vec X, Vec Y):
        print("ERROR: function not implemented.")


