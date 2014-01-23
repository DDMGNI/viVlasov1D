'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as npy
cimport numpy as npy

from petsc4py import PETSc

from vlasov.toolbox.Toolbox import Toolbox


cdef class PETScVlasovSolver(PETScVlasovSolverBase):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 VIDA da1  not None,
                 VIDA dax  not None,
                 VIDA day  not None,
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
                 regularisation=0.):
        '''
        Constructor
        '''
        
        super().__init__(da1, grid, H0, H1p, H1h, H2p, H1h, charge, coll_freq, coll_diff, coll_drag, regularisation)
        
        # distributed arrays
        self.dax = dax
        self.day = day
        
        # interim vectors
        self.X = self.da1.createGlobalVec()     # LHS
        self.B = self.da1.createGlobalVec()     # RHS
        self.F = self.da1.createGlobalVec()     # FFT
        
        # get local index ranges
        (xs, xe), = self.dax.getRanges()
        (ys, ye), = self.day.getRanges()
        
        
        # x and v vectors
        self.xvecs = {}
        self.yvecs = {}
        
        for i in range(ys, ye):
            self.xvecs[i] = PETSc.Vec().createSeq(self.grid.nx)

        for i in range(0, self.grid.nx):
            self.yvecs[i] = self.day.createGlobalVec()
        
        
        # eigenvectors
        lambdas = npy.empty(self.nx, dtype=npy.complex128)
        
        for i in range(0, self.grid.nx):
            lambdas[i] = npy.exp(2.j * npy.pi * float(i) / self.grid.nx) \
                       - npy.exp(2.j * npy.pi * float(i) / self.grid.nx * (self.grid.nx-1)) \
        
        
        # prototype matrix
        proto = PETSc.Mat().create()
        proto.setType(PETSc.Mat.MatType.SEQAIJ)
        proto.setSizes([self.grid.nv**2, self.grid.nv**2])
        proto.setUp()
        
        self.formPreconditionerMatrix(proto)
        
        
        # identity matrix
        identity = PETSc.Mat().create()
        identity.setType(PETSc.Mat.MatType.SEQAIJ)
        identity.setSizes([self.grid.nv**2, self.grid.nv**2])
        identity.setUp()
        
        identity.zeroEntries()
        
        for i in range(0, self.grid.nv): identity.setValue(i, i, 1. * self.grid.ht_inv)
        
        identity.assemble()
        
        
        # preconditioner matrices
        self.pmats = {}
        
        for i in range(ys, ye):
            self.pmats[i] = PETSc.Mat().create()
            self.pmats[i].setType(PETSc.Mat.MatType.SEQAIJ)
            self.pmats[i].setSizes([self.grid.nv**2, self.grid.nv**2])
            self.pmats[i].setUp()
            
            identity.copy(self.pmats[i])
            self.pmats[i].axpy(lambdas[i], proto)
        
        
        # destroy temporary matrices 
        proto.destroy()
        identity.destroy()
        
        
        # fftw matrix
        fvec = self.day.createGlobalVec()
        
        fftw = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        fftw.setType(PETSc.Mat.MatType.FFTW)
        fftw.setSizes([fvec.getSizes(), fvec.getSizes()])
        fftw.setUp()
        
        
            
    
    def jacobian(self, Vec F, Vec Y):
        self.jacobianArakawaJ4(F, self.X)
        self.tensorProduct(self.X, Y)
    
    
    def function(self, Vec F, Vec Y):
        self.functionArakawaJ4(F, self.B)
        self.tensorProduct(self.B, Y)
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def tensorProduct(self, Vec X, Vec Y):
        pass
    
        
        
#         cdef npy.int64_t i, j
#         cdef npy.int64_t ix, iy, jx, jy
#         cdef npy.int64_t xe, xs, ye, ys
#         
#         
#         cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
#         cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
#         cdef npy.float64_t result_J1, result_J2, result_J4
#         cdef npy.float64_t coll_drag, coll_diff
#         
#         self.get_data_arrays()
#         
#         (xs, xe), (ys, ye) = self.da1.getRanges()
#         
#         cdef npy.ndarray[npy.float64_t, ndim=2] fp = self.da1.getLocalArray(F, self.localFp)
#         cdef npy.ndarray[npy.float64_t, ndim=2] y  = self.da1.getGlobalArray(Y)
#         
#         cdef npy.ndarray[npy.float64_t, ndim=2] fh    = self.fh
#         cdef npy.ndarray[npy.float64_t, ndim=2] f_ave = 0.5 * (fp + fh)
#         cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1p + self.h1h) \
#                                                                 + 0.5 * (self.h2p + self.h2h)
#         
#         
#         
#         for i in range(xs, xe):
#             ix = i-xs+self.grid.stencil
#             iy = i-xs
#             
#             # Vlasov equation
#             for j in range(ys, ye):
#                 jx = j-ys+self.grid.stencil
#                 jy = j-ys

    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def formPreconditionerMatrix(self, Mat A):
        cdef npy.int64_t i, j
        
        cdef npy.ndarray[npy.float64_t, ndim=1] v = self.grid.v
        
        cdef npy.float64_t arak_fac_J1   = 0.5 / (12. * self.grid.hx * self.grid.hv)
        
#         cdef npy.float64_t time_fac      = 1.0  / self.grid.ht
#         cdef npy.float64_t arak_fac_J1   = + 1.0 / (12. * self.grid.hx * self.grid.hv)
#         cdef npy.float64_t arak_fac_J2   = - 0.5 / (24. * self.grid.hx * self.grid.hv)
        
        
        A.zeroEntries()
        
        for i in range(0, self.grid.nv):
        
#             for index, value in [
#                     ((i-2, j  ), - (h[ix, jx+1] - h[ix, jx-1]) * arak_fac_J2),
#                     ((i-1, j-1), - (h[ix, jx  ] - h[ix, jx-1]) * arak_fac_J1 \
#                                  - (h[ix, jx  ] - h[ix, jx-2]) * arak_fac_J2 \
#                                  - (h[ix, jx+1] - h[ix, jx-1]) * arak_fac_J2),
#                     ((i-1, j  ), - (h[ix, jx+1] - h[ix, jx-1]) * arak_fac_J1 \
#                                  - (h[ix, jx+1] - h[ix, jx-1]) * arak_fac_J1),
#                     ((i-1, j+1), - (h[ix, jx+1] - h[ix, jx  ]) * arak_fac_J1 \
#                                  - (h[ix, jx+2] - h[ix, jx  ]) * arak_fac_J2 \
#                                  - (h[ix, jx+1] - h[ix, jx-1]) * arak_fac_J2),
#                     ((i,   j  ), + time_fac),
#                     ((i+1, j-1), + (h[ix, jx  ] - h[ix, jx-1]) * arak_fac_J1 \
#                                  + (h[ix, jx  ] - h[ix, jx-2]) * arak_fac_J2 \
#                                  + (h[ix, jx+1] - h[ix, jx-1]) * arak_fac_J2),
#                     ((i+1, j  ), + (h[ix, jx+1] - h[ix, jx-1]) * arak_fac_J1 \
#                                  + (h[ix, jx+1] - h[ix, jx-1]) * arak_fac_J1),
#                     ((i+1, j+1), + (h[ix, jx+1] - h[ix, jx  ]) * arak_fac_J1 \
#                                  + (h[ix, jx+2] - h[ix, jx  ]) * arak_fac_J2 \
#                                  + (h[ix, jx+1] - h[ix, jx-1]) * arak_fac_J2),
#                     ((i+2, j  ), + (h[ix, jx+1] - h[ix, jx-1]) * arak_fac_J2),
#                 ]:

            for j, value in [
                    (i-1, - 0.5 * (v[j  ]**2 - v[j-1]**2) * arak_fac_J1),
                    (i,   - 1.0 * (v[j+1]**2 - v[j-1]**2) * arak_fac_J1),
                    (i+1, - 0.5 * (v[j+1]**2 - v[j  ]**2) * arak_fac_J1),
                ]:
                
                A.setValue(i, j, value)
                        
        A.assemble()
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobianArakawaJ4(self, Vec F, Vec Y):
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
    def functionArakawaJ4(self, Vec F, Vec Y):
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
