'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as npy
cimport numpy as npy

from scipy.sparse        import diags
from scipy.sparse.linalg import splu
from scipy.fftpack       import fft, ifft, fftshift, ifftshift

from petsc4py import PETSc


cdef class PETScVlasovSolver(PETScVlasovPreconditioner):
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
        
        # temporary arrays
        (xs, xe), (ys, ye) = self.day.getRanges()
        self.bsolver = npy.empty((xe-xs, self.grid.nv), dtype=npy.complex128)
                                 
        (xs, xe), (ys, ye) = self.dax.getRanges()
        self.tfft    = npy.empty((self.grid.nx, ye-ys), dtype=npy.complex128)
        
        
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
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef solve(self, Vec XR, Vec XI, Vec YR, Vec YI):
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
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef jacobianSolver(self, Vec F, Vec Y):
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
    cdef functionSolver(self, Vec F, Vec Y):
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
