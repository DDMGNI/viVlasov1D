# cython: profile=True
'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as npy
cimport numpy as npy

from scipy.sparse        import diags, eye
from scipy.sparse.linalg import splu
from scipy.fftpack       import fftshift, ifftshift

# from scipy.fftpack                   import fft, ifft
# from scipy.fftpack                   import rfft, irfft
import pyfftw
from pyfftw.interfaces.scipy_fftpack import fft, ifft
# from pyfftw.interfaces.scipy_fftpack import rfft, irfft

from petsc4py import PETSc

from vlasov.toolbox.Toolbox import Toolbox

# import vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4


cdef class PETScVlasovSolver(PETScVlasovSolverBase):
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
        
#         # wrapped solver
#         self.solver = vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4.PETScVlasovSolver(da1, grid, H0, H1p, H1h, H2p, H2h, charge, coll_freq, coll_diff, coll_drag, regularisation)
        
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
        
        (self.dax_xs, self.dax_xe), (self.dax_ys, self.dax_ye) = self.dax.getRanges()
        
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
        
        
        # temporary arrays
        (xs, xe), (ys, ye) = self.day.getRanges()
        self.bsolver = npy.empty((xe-xs, self.grid.nv), dtype=npy.complex128)
                                 
        (xs, xe), (ys, ye) = self.dax.getRanges()
        self.tfft    = npy.empty((self.grid.nx, ye-ys), dtype=npy.complex128)
        
        
        # FFTW arrays 
        self.fftw_in   = pyfftw.n_byte_align_empty((ye-ys, self.grid.nx), 16, 'complex128')
        self.fftw_out  = pyfftw.n_byte_align_empty((ye-ys, self.grid.nx), 16, 'complex128')
        
        self.ifftw_in  = pyfftw.n_byte_align_empty((ye-ys, self.grid.nx), 16, 'complex128')
        self.ifftw_out = pyfftw.n_byte_align_empty((ye-ys, self.grid.nx), 16, 'complex128')
        
        # enable cache in pyFFTW for optimal performance
        pyfftw.interfaces.cache.enable()
        
        # create pyFFTW plans
        self.fftw_plan  = pyfftw.FFTW(self.fftw_in,  self.fftw_out,  axes=(1,), direction='FFTW_FORWARD')
        self.ifftw_plan = pyfftw.FFTW(self.ifftw_in, self.ifftw_out, axes=(1,), direction='FFTW_BACKWARD')
        
        
        # get local x ranges for solver
        (xs, xe), (ys, ye) = self.day.getRanges()
        
        # eigenvalues
        eigen = npy.empty(self.grid.nx, dtype=npy.complex128)
        
        for i in range(0, self.grid.nx):
            eigen[i] = npy.exp(2.j * npy.pi * float(i) / self.grid.nx * (self.grid.nx-1)) \
                     - npy.exp(2.j * npy.pi * float(i) / self.grid.nx)
        
        eigen[:] = ifftshift(eigen)
        
        # LU decompositions
        self.solvers = [splu(self.formSparsePreconditionerMatrix(eigen[i+xs])) for i in range(0, xe-xs)]
        
        
        # LAPACK parameters
        self.M = self.grid.nv
        self.N = self.grid.nv
        self.KL = 1
        self.KU = 1
        self.NRHS = 1
        self.LDA  = 4
        self.LDB  = self.grid.nv
        self.T = 'N'
        
        # matrices, rhs, pivots
        self.matrices = npy.zeros((4, self.grid.nv, xe-xs), dtype=npy.cdouble, order='F')
        self.rhs      = npy.empty((1, self.grid.nv, xe-xs), dtype=npy.cdouble, order='F')
        self.pivots   = npy.empty((self.grid.nv, xe-xs), dtype=npy.int64, order='F')
        
        # build matrices
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Creating Preconditioner Matrices.")
         
        for i in range(0, xe-xs):
            self.formBandedPreconditionerMatrix(self.matrices[:,:,i], eigen[i+xs])
         
        # LU decompositions
        if PETSc.COMM_WORLD.getRank() == 0:
            print("LU Decomposing Preconditioner Matrices.")
         
        for i in range(0, xe-xs):
            if self.call_zgbtrf(self.matrices[:,:,i], self.pivots[:,i]) != 0:
                print("   ERROR in LU Decomposition.")
         
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Preconditioner Initialisation done.")
        
    
    cpdef jacobian(self, Vec F, Vec Y):
        self.jacobianArakawaJ4(F, self.X)
        self.tensorProduct(self.X, Y)
    
    
    cpdef function(self, Vec F, Vec Y):
        self.functionArakawaJ4(F, self.B)
        self.tensorProduct(self.B, Y)
        

    cdef tensorProduct(self, Vec X, Vec Y):
        
        self.copy_da1_to_dax(X, self.F)                  # copy X(da1) to F(dax)
        
        self.fftw(self.F, self.FfftR, self.FfftI)        # FFT F for each v
        
        self.copy_dax_to_day(self.FfftR, self.BfftR)     # copy F'(dax) to B'(day)
        self.copy_dax_to_day(self.FfftI, self.BfftI)     # copy F'(dax) to B'(day)
        
#         self.solve_scipy(self.BfftR, self.BfftI,
#                          self.CfftR, self.CfftI)         # solve AC'=B' for each x
        
        self.solve_lapack(self.BfftR, self.BfftI,
                          self.CfftR, self.CfftI)        # solve AC'=B' for each x
        
#         self.BfftR.copy(self.CfftR)
#         self.BfftI.copy(self.CfftI)
        
        self.copy_day_to_dax(self.CfftR, self.ZfftR)     # copy C'(day) to Z'(dax)
        self.copy_day_to_dax(self.CfftI, self.ZfftI)     # copy C'(day) to Z'(dax)
        
        self.ifftw(self.ZfftR, self.ZfftI, self.Z)       # iFFT Z' for each v
        
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
        
    
    cdef fft(self, Vec X, Vec YR, Vec YI):
        # Fourier Transform for each v
        
        cdef npy.uint64_t i, j
        cdef npy.uint64_t n1, n2, n3
        cdef npy.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        assert xs == 0
        assert xe == self.grid.nx
        
        cdef npy.ndarray[npy.float64_t, ndim=2] x  = self.dax.getGlobalArray(X)
        cdef npy.ndarray[npy.float64_t, ndim=2] yr = self.dax.getGlobalArray(YR)
        cdef npy.ndarray[npy.float64_t, ndim=2] yi = self.dax.getGlobalArray(YI)
        
        cdef npy.ndarray[npy.complex128_t, ndim=2] z
        
        z = fft(x, axis=0)
        
        yr[:,:] = z.real
        yi[:,:] = z.imag
        
    
    cdef ifft(self, Vec XR, Vec XI, Vec Y):
        # inverse Fourier Transform for each v
        
        cdef npy.uint64_t i, j
        cdef npy.uint64_t n1, n2, n3
        cdef npy.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        assert xs == 0
        assert xe == self.grid.nx
        
        cdef npy.ndarray[npy.float64_t, ndim=2] xr = self.dax.getGlobalArray(XR)
        cdef npy.ndarray[npy.float64_t, ndim=2] xi = self.dax.getGlobalArray(XI)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.dax.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.complex128_t, ndim=2] z = self.tfft
        
        z[:,:].real = xr
        z[:,:].imag = xi
        
        y[:,:] = ifft(z, axis=0).real
        
    
    cdef fftw(self, Vec X, Vec YR, Vec YI):
        # Fourier Transform for each v
        
        shape = (self.dax_ye-self.dax_ys, self.dax_xe-self.dax_xs)
         
        cdef npy.ndarray[npy.float64_t, ndim=2] yr = YR.getArray().reshape(shape, order='c')
        cdef npy.ndarray[npy.float64_t, ndim=2] yi = YI.getArray().reshape(shape, order='c')

        cdef npy.ndarray[npy.complex128_t, ndim=2] x = self.fftw_in
        cdef npy.ndarray[npy.complex128_t, ndim=2] y = self.fftw_out
        
        x[:,:].real = X.getArray().reshape(shape, order='c')
#         x[:,:] = X.getArray().reshape(shape, order='c').astype(npy.float64).view(npy.complex128)
        
        self.fftw_plan()

        yr[:,:] = y.real
        yi[:,:] = y.imag
        
    
    cdef ifftw(self, Vec XR, Vec XI, Vec Y):
        # inverse Fourier Transform for each v
        
        shape = (self.dax_ye-self.dax_ys, self.dax_xe-self.dax_xs)
         
        cdef npy.ndarray[npy.float64_t, ndim=2] xr = XR.getArray().reshape(shape, order='c')
        cdef npy.ndarray[npy.float64_t, ndim=2] xi = XI.getArray().reshape(shape, order='c')
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = Y.getArray().reshape(shape, order='c')
        
        cdef npy.ndarray[npy.complex128_t, ndim=2] x = self.ifftw_in
        cdef npy.ndarray[npy.complex128_t, ndim=2] z = self.ifftw_out
        
        x[:,:].real = xr
        x[:,:].imag = xi
        
        self.ifftw_plan()
        
        y[:,:] = z.real
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef solve_scipy(self, Vec XR, Vec XI, Vec YR, Vec YI):
        # solve system for each x
        
        cdef npy.int64_t i, j
        cdef npy.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.day.getRanges()
        
        assert ys == 0
        assert ye == self.grid.nv
        
        cdef npy.ndarray[npy.float64_t, ndim=2] xr = self.day.getGlobalArray(XR)
        cdef npy.ndarray[npy.float64_t, ndim=2] xi = self.day.getGlobalArray(XI)
        cdef npy.ndarray[npy.float64_t, ndim=2] yr = self.day.getGlobalArray(YR)
        cdef npy.ndarray[npy.float64_t, ndim=2] yi = self.day.getGlobalArray(YI)
        
        cdef npy.ndarray[npy.complex128_t, ndim=2] b = self.bsolver
        cdef npy.ndarray[npy.complex128_t, ndim=1] c
        
        b[:,:].real = xr
        b[:,:].imag = xi
        
        for i in range(0, xe-xs):
            c = self.solvers[i].solve(b[i,:])
            
            yr[i,:] = c.real
            yi[i,:] = c.imag
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef solve_lapack(self, Vec XR, Vec XI, Vec YR, Vec YI):
        # solve system for each x
        
        cdef npy.int64_t i, j
        cdef npy.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.day.getRanges()
        
        assert ys == 0
        assert ye == self.grid.nv
        
        shape = (ye-ys, xe-xs)
         
        cdef npy.ndarray[npy.float64_t, ndim=2] xr = XR.getArray().reshape(shape, order='c')
        cdef npy.ndarray[npy.float64_t, ndim=2] xi = XI.getArray().reshape(shape, order='c')
        cdef npy.ndarray[npy.float64_t, ndim=2] yr = YR.getArray().reshape(shape, order='c')
        cdef npy.ndarray[npy.float64_t, ndim=2] yi = YI.getArray().reshape(shape, order='c')
        
        cdef npy.ndarray[npy.complex128_t, ndim=3] a = self.matrices
        cdef npy.ndarray[npy.complex128_t, ndim=3] b = self.rhs
        cdef npy.ndarray[npy.int64_t, ndim=2] p = self.pivots
        
        b[0,:,:].real = xr
        b[0,:,:].imag = xi
        
        for i in range(0, xe-xs):
            self.call_zgbtrs(a[:,:,i], b[:,:,i], p[:,i])
        
        yr[:,:] = b[0].real
        yi[:,:] = b[0].imag
        
    
    cdef call_zgbtrf(self, npy.ndarray matrix, npy.ndarray pivots):
        cdef int INFO = 0
          
        zgbtrf(&self.M, &self.N, &self.KL, &self.KU, <complex*>matrix.data, &self.LDA, <int*>pivots.data, &INFO)
        
        return INFO
     
 
    cdef call_zgbtrs(self, npy.ndarray[npy.complex128_t, ndim=2] matrix,
                           npy.ndarray[npy.complex128_t, ndim=2] rhs,
                           npy.ndarray[npy.int64_t, ndim=1] pivots):
        
        cdef int INFO = 0
         
        zgbtrs(&self.T, &self.N, &self.KL, &self.KU, &self.NRHS, <complex*>matrix.data, &self.LDA, <int*>pivots.data, <complex*>rhs.data, &self.LDB, &INFO)
        
        return INFO
        
    
    cdef formSparsePreconditionerMatrix(self, npy.complex eigen):
        cdef npy.int64_t j
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v = self.grid.v
        
        cdef npy.float64_t arak_fac_J1 = 0.5 / (12. * self.grid.hx * self.grid.hv)
        
        diagm = npy.zeros(self.grid.nv, dtype=npy.complex128)
        diag  = npy.ones (self.grid.nv, dtype=npy.complex128)
        diagp = npy.zeros(self.grid.nv, dtype=npy.complex128)
        
        for j in range(2, self.grid.nv-2):
            diagm[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] - self.grid.hv2 ) * arak_fac_J1
            diag [j] = eigen * 4.0 * self.grid.hv * v[j] * arak_fac_J1 + self.grid.ht_inv
            diagp[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] + self.grid.hv2 ) * arak_fac_J1
        
        offsets   = [-1, 0, +1]
        diagonals = [diagm[1:], diag, diagp[:-1]]
        
        return diags(diagonals, offsets, shape=(self.grid.nv, self.grid.nv), format='csc', dtype=npy.complex128)
        
    
    cdef formBandedPreconditionerMatrix(self, npy.ndarray matrix, npy.complex eigen):
        cdef npy.int64_t j
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v = self.grid.v
        
        cdef npy.float64_t arak_fac_J1 = 0.5 / (12. * self.grid.hx * self.grid.hv)
        
        
        diagm = npy.zeros(self.grid.nv, dtype=npy.cdouble)
        diag  = npy.ones (self.grid.nv, dtype=npy.cdouble)
        diagp = npy.zeros(self.grid.nv, dtype=npy.cdouble)
        
        for j in range(2, self.grid.nv-2):
            diagm[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] - self.grid.hv2 ) * arak_fac_J1
            diag [j] = eigen * 4.0 * self.grid.hv * v[j] * arak_fac_J1 + self.grid.ht_inv
            diagp[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] + self.grid.hv2 ) * arak_fac_J1
        
        matrix[1, 1:  ] = diagp[:-1]
        matrix[2,  :  ] = diag
        matrix[3,  :-1] = diagm[1:]
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef jacobianArakawaJ1(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        cdef npy.float64_t coll_drag, coll_diff
        
        self.get_data_arrays()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fd = self.da1.getLocalArray(F, self.localFd)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v     = self.grid.v
        cdef npy.ndarray[npy.float64_t, ndim=1] u     = self.up
        cdef npy.ndarray[npy.float64_t, ndim=1] a     = self.ap
        
        
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = fd[ix, jx]
                     
                else:
                    # Arakawa's J1
                    jpp_J1 = (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                           - (fd[ix,   jx+1] - fd[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                    
                    jpc_J1 = fd[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - fd[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - fd[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + fd[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J1 = fd[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                           - fd[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                           - fd[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                           + fd[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
         
                    # collision operator
                    coll_drag = ( (v[j+1] - u[ix]) * fd[ix, jx+1] - (v[j-1] - u[ix]) * fd[ix, jx-1] ) * a[ix]
                    coll_diff = ( fd[ix, jx+1] - 2. * fd[ix, jx] + fd[ix, jx-1] )
                    
         
                    y[iy, jy] = fd[ix, jx] * self.grid.ht_inv \
                             + 0.5 * result_J1 * self.grid.hx_inv * self.grid.hv_inv \
                             - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                             - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef functionArakawaJ1(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        cdef npy.float64_t coll_drag, coll_diff
        
        self.get_data_arrays()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fp = self.da1.getLocalArray(F, self.localFp)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fh    = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v     = self.grid.v
        cdef npy.ndarray[npy.float64_t, ndim=1] up    = self.up
        cdef npy.ndarray[npy.float64_t, ndim=1] ap    = self.ap
        cdef npy.ndarray[npy.float64_t, ndim=1] uh    = self.uh
        cdef npy.ndarray[npy.float64_t, ndim=1] ah    = self.ah
        
        
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = fp[ix, jx]
                     
                else:
                    # Arakawa's J1
                    jpp_J1 = (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                           - (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                     
                    jpc_J1 = f_ave[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - f_ave[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - f_ave[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + f_ave[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                     
                    jcp_J1 = f_ave[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                           - f_ave[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                           - f_ave[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                           + f_ave[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                     
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                     
                    # collision operator
                    coll_drag = ( (v[j+1] - up[ix]) * fp[ix, jx+1] - (v[j-1] - up[ix]) * fp[ix, jx-1] ) * ap[ix] \
                              + ( (v[j+1] - uh[ix]) * fh[ix, jx+1] - (v[j-1] - uh[ix]) * fh[ix, jx-1] ) * ah[ix]
                    coll_diff = ( fp[ix, jx+1] - 2. * fp[ix, jx] + fp[ix, jx-1] ) \
                              + ( fh[ix, jx+1] - 2. * fh[ix, jx] + fh[ix, jx-1] )
                     
                     
                    y[iy, jy] = (fp[ix, jx] - fh[ix, jx]) * self.grid.ht_inv \
                             + result_J1 * self.grid.hx_inv * self.grid.hv_inv \
                             - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                             - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef jacobianArakawaJ4(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4, poisson
        cdef npy.float64_t coll_drag, coll_diff
        cdef npy.float64_t collisions = 0.
        cdef npy.float64_t regularisation = 0.
        
        self.get_data_arrays()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fd = self.da1.getLocalArray(F, self.localFd)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v     = self.grid.v
        cdef npy.ndarray[npy.float64_t, ndim=1] u     = self.up
        cdef npy.ndarray[npy.float64_t, ndim=1] a     = self.ap
        
        
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = fd[ix, jx]
                    
                else:
                    # Arakawa's J1
                    jpp_J1 = (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                           - (fd[ix,   jx+1] - fd[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                    
                    jpc_J1 = fd[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - fd[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - fd[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + fd[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J1 = fd[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                           - fd[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                           - fd[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                           + fd[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                    
                    # Arakawa's J2
                    jcc_J2 = (fd[ix+1, jx+1] - fd[ix-1, jx-1]) * (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) \
                           - (fd[ix-1, jx+1] - fd[ix+1, jx-1]) * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1])
                    
                    jpc_J2 = fd[ix+2, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - fd[ix-2, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - fd[ix,   jx+2] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + fd[ix,   jx-2] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J2 = fd[ix+1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) \
                           - fd[ix-1, jx-1] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx-2]) \
                           - fd[ix-1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix-2, jx  ]) \
                           + fd[ix+1, jx-1] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    poisson   = 0.5 * result_J4 * self.grid.hx_inv * self.grid.hv_inv \
                    
                    
                    # collision operator
                    if self.nu > 0.:
                        coll_drag = ( (v[j+1] - u[ix]) * fd[ix, jx+1] - (v[j-1] - u[ix]) * fd[ix, jx-1] ) * a[ix]
                        
                        coll_diff = ( fd[ix, jx+1] - 2. * fd[ix, jx] + fd[ix, jx-1] )
                        
                        collisions = \
                              - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                              - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv \
                    
                    # regularisation
                    if self.regularisation != 0.:
                        regularisation = \
                              + self.grid.ht * self.regularisation * self.grid.hx2_inv * ( 2. * fd[ix, jx] - fd[ix+1, jx] - fd[ix-1, jx] ) \
                              + self.grid.ht * self.regularisation * self.grid.hv2_inv * ( 2. * fd[ix, jx] - fd[ix, jx+1] - fd[ix, jx-1] )
                    
                    # solution
                    y[iy, jy] = fd[ix, jx] * self.grid.ht_inv \
                              + poisson \
                              + collisions \
                              + regularisation
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef functionArakawaJ4(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4, poisson
        cdef npy.float64_t coll_drag, coll_diff
        cdef npy.float64_t collisions = 0.
        cdef npy.float64_t regularisation = 0.
        
        self.get_data_arrays()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fp = self.da1.getLocalArray(F, self.localFp)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] fh    = self.fh
        cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
        cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
                                                                + 0.5 * (self.h2p + self.h2h)
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v     = self.grid.v
        cdef npy.ndarray[npy.float64_t, ndim=1] up    = self.up
        cdef npy.ndarray[npy.float64_t, ndim=1] ap    = self.ap
        cdef npy.ndarray[npy.float64_t, ndim=1] uh    = self.uh
        cdef npy.ndarray[npy.float64_t, ndim=1] ah    = self.ah
        
        
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = fp[ix, jx]
                    
                else:
                    # Arakawa's J1
                    jpp_J1 = (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                           - (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) * (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ])
                    
                    jpc_J1 = f_ave[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - f_ave[ix-1, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - f_ave[ix,   jx+1] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + f_ave[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J1 = f_ave[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                           - f_ave[ix-1, jx-1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) \
                           - f_ave[ix-1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) \
                           + f_ave[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1])
                    
                    # Arakawa's J2
                    jcc_J2 = (f_ave[ix+1, jx+1] - f_ave[ix-1, jx-1]) * (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) \
                           - (f_ave[ix-1, jx+1] - f_ave[ix+1, jx-1]) * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1])
                    
                    jpc_J2 = f_ave[ix+2, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                           - f_ave[ix-2, jx  ] * (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) \
                           - f_ave[ix,   jx+2] * (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) \
                           + f_ave[ix,   jx-2] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1])
                    
                    jcp_J2 = f_ave[ix+1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) \
                           - f_ave[ix-1, jx-1] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx-2]) \
                           - f_ave[ix-1, jx+1] * (h_ave[ix,   jx+2] - h_ave[ix-2, jx  ]) \
                           + f_ave[ix+1, jx-1] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    poisson   = result_J4 * self.grid.hx_inv * self.grid.hv_inv
                    
                    
                    # collision operator
                    if self.nu > 0.:
                        coll_drag = ( (v[j+1] - up[ix]) * fp[ix, jx+1] - (v[j-1] - up[ix]) * fp[ix, jx-1] ) * ap[ix] \
                                  + ( (v[j+1] - uh[ix]) * fh[ix, jx+1] - (v[j-1] - uh[ix]) * fh[ix, jx-1] ) * ah[ix]
                        
                        coll_diff = ( fp[ix, jx+1] - 2. * fp[ix, jx] + fp[ix, jx-1] ) \
                                  + ( fh[ix, jx+1] - 2. * fh[ix, jx] + fh[ix, jx-1] )
                        
                        collisions = \
                                   - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                                   - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv \
                    
                    # regularisation
                    if self.regularisation != 0.:
                        regularisation = \
                                       + self.grid.ht * self.regularisation * self.grid.hx2_inv * ( 2. * fp[ix, jx] - fp[ix+1, jx] - fp[ix-1, jx] ) \
                                       + self.grid.ht * self.regularisation * self.grid.hv2_inv * ( 2. * fp[ix, jx] - fp[ix, jx+1] - fp[ix, jx-1] )
                    
                    # solution
                    y[iy, jy] = (fp[ix, jx] - fh[ix, jx]) * self.grid.ht_inv \
                              + poisson \
                              + collisions \
                              + regularisation
