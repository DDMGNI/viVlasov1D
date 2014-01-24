# cython: profile=True
'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as npy
cimport numpy as npy

from petsc4py import PETSc


cdef class PETScVlasovPreconditioner(PETScVlasovSolverBase):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 VIDA da1  not None,
                 Grid grid not None,
                 Vec H0  not None,
                 Vec H1p not None,
                 Vec H1h not None,
                 Vec H2p not None,
                 Vec H2h not None,
                 npy.float64_t charge=-1.,
                 npy.float64_t coll_freq=0.,
                 npy.float64_t coll_diff=1.,
                 npy.float64_t coll_drag=1.,
                 npy.float64_t regularisation=0.):
        '''
        Constructor
        '''
        
        super().__init__(da1, grid, H0, H1p, H1h, H2p, H2h, charge, coll_freq, coll_diff, coll_drag, regularisation)
        
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
        
        # interim vectors
        self.B     = self.da1.createGlobalVec()
        self.X     = self.da1.createGlobalVec()
        self.F     = self.dax.createGlobalVec()
        self.FfftR = self.dax.createGlobalVec()
        self.FfftI = self.dax.createGlobalVec()
        self.BfftR = self.day.createGlobalVec()
        self.BfftI = self.day.createGlobalVec()
        self.CfftR = self.day.createGlobalVec()
        self.CfftI = self.day.createGlobalVec()
        self.ZfftR = self.dax.createGlobalVec()
        self.ZfftI = self.dax.createGlobalVec()
        self.Z     = self.dax.createGlobalVec()
        
        
    
    cpdef jacobian(self, Vec F, Vec Y):
        self.jacobianSolver(F, self.X)
        self.tensorProduct(self.X, Y)
    
    
    cpdef function(self, Vec F, Vec Y):
        self.functionSolver(F, self.B)
        self.tensorProduct(self.B, Y)
        

    cdef tensorProduct(self, Vec X, Vec Y):
        
        self.copy_da1_to_dax(X, self.F)                  # copy X(da1) to F(dax)
        
        self.fft(self.F, self.FfftR, self.FfftI)         # FFT F for each v
        
        self.copy_dax_to_day(self.FfftR, self.BfftR)     # copy F'(dax) to B'(day)
        self.copy_dax_to_day(self.FfftI, self.BfftI)     # copy F'(dax) to B'(day)
        
        self.solve(self.BfftR, self.BfftI,
                   self.CfftR, self.CfftI)               # solve AC'=B' for each x
        
        self.copy_day_to_dax(self.CfftR, self.ZfftR)     # copy C'(day) to Z'(dax)
        self.copy_day_to_dax(self.CfftI, self.ZfftI)     # copy C'(day) to Z'(dax)
        
        self.ifft(self.ZfftR, self.ZfftI, self.Z)        # iFFT Z' for each v
        
        self.copy_dax_to_da1(self.Z, Y)                  # copy Z(dax) to Y(da1)
        
    
    cdef copy_da1_to_dax(self, Vec X, Vec Y):
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        (xsx, xex), (ysx, yex) = self.dax.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] x
        cdef npy.ndarray[npy.float64_t, ndim=2] y
        
        if xs1 == xsx and xe1 == xex and ys1 == ysx and ye1 == yex:
            x = self.da1.getGlobalArray(X)
            y = self.dax.getGlobalArray(Y)
            y[:,:] = x[:,:]
        else:
            aox = self.dax.getAO()
            ao1 = self.da1.getAO()
            
            appindices = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            xpindices  = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            ypindices  = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            
            aox.app2petsc(xpindices)
            ao1.app2petsc(ypindices)
            
            scatter = PETSc.Scatter().create(X, ypindices, Y, xpindices)
            
            scatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            
            scatter.destroy()
         
        
        
    cdef copy_dax_to_da1(self, Vec X, Vec Y):
        (xsx, xex), (ysx, yex) = self.dax.getRanges()
        (xs1, xe1), (ys1, ye1) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] x
        cdef npy.ndarray[npy.float64_t, ndim=2] y
        
        if xs1 == xsx and xe1 == xex and ys1 == ysx and ye1 == yex:
            x = self.dax.getGlobalArray(X)
            y = self.da1.getGlobalArray(Y)
            y[:,:] = x[:,:]
        else:
            aox = self.dax.getAO()
            ao1 = self.da1.getAO()
            
            appindices = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            xpindices  = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            ypindices  = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            
            aox.app2petsc(xpindices)
            ao1.app2petsc(ypindices)
            
            scatter = PETSc.Scatter().create(X, xpindices, Y, ypindices)
            
            scatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            
            scatter.destroy()
        
        
    cdef copy_dax_to_day(self, Vec X, Vec Y):
        (xsx, xex), (ysx, yex) = self.dax.getRanges()
        (xsy, xey), (ysy, yey) = self.day.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] x
        cdef npy.ndarray[npy.float64_t, ndim=2] y
        
        if xsy == xsx and xey == xex and ysy == ysx and yey == yex:
            x = self.dax.getGlobalArray(X)
            y = self.day.getGlobalArray(Y)
            y[:,:] = x[:,:]
            
        else:
            aox = self.dax.getAO()
            aoy = self.day.getAO()
            
            appindices = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            xpindices  = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            ypindices  = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            
            aox.app2petsc(xpindices)
            aoy.app2petsc(ypindices)
        
            scatter = PETSc.Scatter().create(X, xpindices, Y, ypindices)
            
            scatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            
            scatter.destroy()
        
    
    cdef copy_day_to_dax(self, Vec X, Vec Y):
        (xsx, xex), (ysx, yex) = self.dax.getRanges()
        (xsy, xey), (ysy, yey) = self.day.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] x
        cdef npy.ndarray[npy.float64_t, ndim=2] y
        
        if xsy == xsx and xey == xex and ysy == ysx and yey == yex:
            x = self.day.getGlobalArray(X)
            y = self.dax.getGlobalArray(Y)
            y[:,:] = x[:,:]
            
        else:
            aox = self.dax.getAO()
            aoy = self.day.getAO()
            
            appindices = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            xpindices  = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            ypindices  = PETSc.IS().createStride((yex-ysx)*self.grid.nx, ysx*self.grid.nx, 1)
            
            aox.app2petsc(xpindices)
            aoy.app2petsc(ypindices)
        
            scatter = PETSc.Scatter().create(X, ypindices, Y, xpindices)
            
            scatter.scatter(X, Y, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            
            scatter.destroy()
        

    cdef fft (self, Vec X, Vec YR, Vec YI):
        print("ERROR: function not implemented.")

    cdef ifft(self, Vec XR, Vec XI, Vec Y):
        print("ERROR: function not implemented.")

    cdef solve(self, Vec XR, Vec XI, Vec YR, Vec YI):
        print("ERROR: function not implemented.")

    cdef jacobianSolver(self, Vec F, Vec Y):
        print("ERROR: function not implemented.")

    cdef functionSolver(self, Vec F, Vec Y):
        print("ERROR: function not implemented.")

