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

from scipy.fftpack                   import rfft, irfft
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
        
        
        # eigenvalues
        eigen = npy.empty(self.grid.nx, dtype=npy.complex128)
        
        for i in range(0, self.grid.nx):
            eigen[i] = npy.exp(2.j * npy.pi * float(i) / self.grid.nx * (self.grid.nx-1)) \
                     - npy.exp(2.j * npy.pi * float(i) / self.grid.nx)
        
        eigen[:] = ifftshift(eigen)
        
        # prototype matrix
        proto = self.formPreconditionerMatrix()
        
        
        # identity matrix
        identity = eye(self.grid.nv, format='csc', dtype=npy.complex128) * self.grid.ht_inv
        
        
        # get local x ranges for solver
        (xs, xe), (ys, ye) = self.day.getRanges()
        
        
        # preconditioner matrices
        self.pmats = {}
         
        for i in range(xs, xe):
            self.pmats[i] = identity + eigen[i] * proto
        
        
        # LU decompositions
        self.solvers = {}
        
        for i in range(xs, xe):
            self.solvers[i] = splu(self.pmats[i])
        
        
    
    def jacobian(self, Vec F, Vec Y):
        self.jacobianArakawaJ4(F, self.X)
        self.tensorProduct(self.X, Y)
    
    
    def function(self, Vec F, Vec Y):
        self.functionArakawaJ4(F, self.B)
        self.tensorProduct(self.B, Y)
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def tensorProduct(self, Vec X, Vec Y):
        
#         hdf5_viewer = PETSc.ViewerHDF5().create("tensor.hdf5",
#                                                 mode=PETSc.Viewer.Mode.WRITE,
#                                                 comm=PETSc.COMM_WORLD)
#         
#         hdf5_viewer.pushGroup("/")
#         
#         # write grid data to hdf5 file
#         X.setName('X')
#         Y.setName('Y')
#         self.F.setName('F')
#         self.Z.setName('Z')
#         self.FfftR.setName('FfftR')
#         self.FfftI.setName('FfftI')
#         self.BfftR.setName('BfftR')
#         self.BfftI.setName('BfftI')
#         self.CfftR.setName('CfftR')
#         self.CfftI.setName('CfftI')
#         self.ZfftR.setName('ZfftR')
#         self.ZfftI.setName('ZfftI')


        self.copy_da1_to_dax(X, self.F)                  # copy X(da1) to F(dax)
        
        self.fft(self.F, self.FfftR, self.FfftI)         # FFT F for each v
        
        self.copy_dax_to_day(self.FfftR, self.BfftR)     # copy F'(dax) to B'(day)
        self.copy_dax_to_day(self.FfftI, self.BfftI)     # copy F'(dax) to B'(day)
        
        self.solve(self.BfftR, self.BfftI,
                   self.CfftR, self.CfftI)               # solve AC'=B' for each x
        
#         self.BfftR.copy(self.CfftR)
#         self.BfftI.copy(self.CfftI)
        
        self.copy_day_to_dax(self.CfftR, self.ZfftR)     # copy C'(day) to Z'(dax)
        self.copy_day_to_dax(self.CfftI, self.ZfftI)     # copy C'(day) to Z'(dax)
        
        self.ifft(self.ZfftR, self.ZfftI, self.Z)        # iFFT Z' for each v
        
        self.copy_dax_to_da1(self.Z, Y)                  # copy Z(dax) to Y(da1)
        
    
#         hdf5_viewer(X)
#         hdf5_viewer(Y)
#         hdf5_viewer(self.F)
#         hdf5_viewer(self.Z)
#         hdf5_viewer(self.FfftR)
#         hdf5_viewer(self.FfftI)
#         hdf5_viewer(self.BfftR)
#         hdf5_viewer(self.BfftI)
#         hdf5_viewer(self.CfftR)
#         hdf5_viewer(self.CfftI)
#         hdf5_viewer(self.ZfftR)
#         hdf5_viewer(self.ZfftI)
        
    
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
        
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        assert xs == 0
        assert xe == self.grid.nx
        
        cdef npy.ndarray[npy.float64_t, ndim=2] x  = self.dax.getGlobalArray(X)
        cdef npy.ndarray[npy.float64_t, ndim=2] yr = self.dax.getGlobalArray(YR)
        cdef npy.ndarray[npy.float64_t, ndim=2] yi = self.dax.getGlobalArray(YI)
        cdef npy.ndarray[npy.float64_t, ndim=2] z  = npy.empty_like(x)
        
        for j in range(0, ye-ys):
            z[:,j] = rfft(x[:,j])
        
        n1 = self.grid.nx
        n2 = int(n1/2)
        n3 = n2+1
        if n1 % 2 != 0:
            n2 += 1
    
        yr[0   , :] = z[0     , :]
        yr[1:n3, :] = z[1:n1:2, :]
        yi[1:n2, :] = z[2:n1:2, :]
    
        for i in range(1, n2):
            yr[-i,:] =  yr[i,:]
            yi[-i,:] = -yi[i,:]
        
    
    cdef ifft(self, Vec XR, Vec XI, Vec Y):
        # inverse Fourier Transform for each v
        
        (xs, xe), (ys, ye) = self.dax.getRanges()
        
        assert xs == 0
        assert xe == self.grid.nx
        
        cdef npy.ndarray[npy.float64_t, ndim=2] xr = self.dax.getGlobalArray(XR)
        cdef npy.ndarray[npy.float64_t, ndim=2] xi = self.dax.getGlobalArray(XI)
        cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.dax.getGlobalArray(Y)
        cdef npy.ndarray[npy.float64_t, ndim=2] z  = npy.empty_like(y)
        
        n1 = self.grid.nx
        n2 = int(n1/2)
        n3 = n2+1
        if n1 % 2 != 0:
            n2 += 1
    
        z[0,      :] = xr[0,    :]
        z[1:n1:2, :] = xr[1:n3, :]
        z[2:n1:2, :] = xi[1:n2, :]
        
        for j in range(0, ye-ys):
            y[:,j] = irfft(z[:,j])
        
    
    cdef solve(self, Vec XR, Vec XI, Vec YR, Vec YI):
        # solve system for each x
        
        cdef npy.int64_t i, j
        cdef npy.int64_t xe, xs, ye, ys
        cdef npy.int64_t n1, n2, n3
        
        (xs, xe), (ys, ye) = self.day.getRanges()
        
        assert ys == 0
        assert ye == self.grid.nv
        
        cdef npy.ndarray[npy.float64_t, ndim=2] xr = self.day.getGlobalArray(XR)
        cdef npy.ndarray[npy.float64_t, ndim=2] xi = self.day.getGlobalArray(XI)
        cdef npy.ndarray[npy.float64_t, ndim=2] yr = self.day.getGlobalArray(YR)
        cdef npy.ndarray[npy.float64_t, ndim=2] yi = self.day.getGlobalArray(YI)
        
        cdef npy.ndarray[npy.complex128_t, ndim=2] b = npy.empty((xe-xs, self.grid.nv), dtype=npy.complex)
        cdef npy.ndarray[npy.complex128_t, ndim=2] c = npy.empty((xe-xs, self.grid.nv), dtype=npy.complex)
        
        b.real[:,:] = xr
        b.imag[:,:] = xi
        
        for i in range(xs, xe):
            c[i-xs,:] = self.solvers[i].solve(b[i-xs,:])
        
        yr[:,:] = c.real
        yi[:,:] = c.imag
    
    
    cdef formPreconditionerMatrix(self):
        cdef npy.int64_t j
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v = self.grid.v
        
        cdef npy.float64_t arak_fac_J1 = 0.5 / (12. * self.grid.hx * self.grid.hv)
        
        
        diagm = npy.zeros(self.grid.nv)
        diag  = npy.ones (self.grid.nv)
        diagp = npy.zeros(self.grid.nv)
        
        for j in range(2, self.grid.nv-2):
            diagm[j] = 0.5 * ( 2. * self.grid.hv * v[j] - self.grid.hv2 ) * arak_fac_J1
            diag [j] = 4.0 * self.grid.hv * v[j] * arak_fac_J1
            diagp[j] = 0.5 * ( 2. * self.grid.hv * v[j] + self.grid.hv2 ) * arak_fac_J1
        
        offsets   = [-1, 0, +1]
        diagonals = [diagm[1:], diag, diagp[:-1]]
        
        return diags(diagonals, offsets, shape=(self.grid.nv, self.grid.nv), format='csc', dtype=npy.complex128)
        
    
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
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
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
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
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
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
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
                    
                    
                    # collision operator
                    coll_drag = ( (v[j+1] - u[ix]) * fd[ix, jx+1] - (v[j-1] - u[ix]) * fd[ix, jx-1] ) * a[ix]
                    coll_diff = ( fd[ix, jx+1] - 2. * fd[ix, jx] + fd[ix, jx-1] )
                    
         
                    y[iy, jy] = fd[ix, jx] * self.grid.ht_inv \
                              + 0.5 * result_J4 * self.grid.hx_inv * self.grid.hv_inv \
                              - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                              - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv \
                              + self.grid.ht * self.regularisation * self.grid.hx2_inv * ( 2. * fd[ix, jx] - fd[ix+1, jx] - fd[ix-1, jx] ) \
                              + self.grid.ht * self.regularisation * self.grid.hv2_inv * ( 2. * fd[ix, jx] - fd[ix, jx+1] - fd[ix, jx-1] )
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef functionArakawaJ4(self, Vec F, Vec Y):
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
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
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
                    
                    
                    # collision operator
                    coll_drag = ( (v[j+1] - up[ix]) * fp[ix, jx+1] - (v[j-1] - up[ix]) * fp[ix, jx-1] ) * ap[ix] \
                              + ( (v[j+1] - uh[ix]) * fh[ix, jx+1] - (v[j-1] - uh[ix]) * fh[ix, jx-1] ) * ah[ix]
                    coll_diff = ( fp[ix, jx+1] - 2. * fp[ix, jx] + fp[ix, jx-1] ) \
                              + ( fh[ix, jx+1] - 2. * fh[ix, jx] + fh[ix, jx-1] )
                    
                    
                    y[iy, jy] = (fp[ix, jx] - fh[ix, jx]) * self.grid.ht_inv \
                              + result_J4 * self.grid.hx_inv * self.grid.hv_inv \
                              - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                              - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv \
                              + self.grid.ht * self.regularisation * self.grid.hx2_inv * ( 2. * fp[ix, jx] - fp[ix+1, jx] - fp[ix-1, jx] ) \
                              + self.grid.ht * self.regularisation * self.grid.hv2_inv * ( 2. * fp[ix, jx] - fp[ix, jx+1] - fp[ix, jx-1] )
