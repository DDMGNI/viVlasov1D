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
    Implements a variational integrator with first order
    finite-difference time-derivative and Arakawa's J4
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
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1p = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localH2p = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()


    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def formJacobian(self, Mat A):
        cdef npy.int64_t i, j, ix, jx
        cdef npy.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[double, ndim=2] h0  = self.da1.getLocalArray(self.H0,  self.localH0)
        cdef npy.ndarray[double, ndim=2] h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
        cdef npy.ndarray[double, ndim=2] h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
        
        cdef double[:,:] hh = h0 + h1h + h2h
        
        
#         cdef npy.float64_t time_fac    = 0.
#         cdef npy.float64_t arak_fac_J1 = 0.
#         cdef npy.float64_t arak_fac_J2 = 0.
#         cdef npy.float64_t coll1_fac   = 0.
#         cdef npy.float64_t coll2_fac   = 0.
        
        cdef npy.float64_t time_fac    = 1.0  / self.grid.ht
        cdef npy.float64_t arak_fac_J1 = + 1.0 / (12. * self.grid.hx * self.grid.hv)
        cdef npy.float64_t arak_fac_J2 = - 0.5 / (24. * self.grid.hx * self.grid.hv)
        
        cdef npy.float64_t coll1_fac   = - 0.5 * self.nu * 0.5 / self.grid.hv
        cdef npy.float64_t coll2_fac   = - 0.5 * self.nu * self.grid.hv2_inv
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        # Vlasov Equation
        for i in range(xs, xe):
            ix = i-xs+2
            
            row.index = (i,)
                
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                row.field = j
                
                # Dirichlet boundary conditions
                if j <= 1 or j >= self.grid.nv-2:
                    A.setValueStencil(row, row, 1.0)
                    
                else:
                    
                    for index, field, value in [
                            ((i-2,), j  , - (hh[ix-1, jx+1] - hh[ix-1, jx-1]) * arak_fac_J2),
                            ((i-1,), j-1, - (hh[ix-1, jx  ] - hh[ix,   jx-1]) * arak_fac_J1 \
                                          - (hh[ix-2, jx  ] - hh[ix,   jx-2]) * arak_fac_J2 \
                                          - (hh[ix-1, jx+1] - hh[ix+1, jx-1]) * arak_fac_J2),
                            ((i-1,), j  , - (hh[ix,   jx+1] - hh[ix,   jx-1]) * arak_fac_J1 \
                                          - (hh[ix-1, jx+1] - hh[ix-1, jx-1]) * arak_fac_J1),
                            ((i-1,), j+1, - (hh[ix,   jx+1] - hh[ix-1, jx  ]) * arak_fac_J1 \
                                          - (hh[ix,   jx+2] - hh[ix-2, jx  ]) * arak_fac_J2 \
                                          - (hh[ix+1, jx+1] - hh[ix-1, jx-1]) * arak_fac_J2),
                            ((i,  ), j-2, + (hh[ix+1, jx-1] - hh[ix-1, jx-1]) * arak_fac_J2),
                            ((i,  ), j-1, + (hh[ix+1, jx  ] - hh[ix-1, jx  ]) * arak_fac_J1 \
                                          + (hh[ix+1, jx-1] - hh[ix-1, jx-1]) * arak_fac_J1 \
                                          - coll1_fac * ( self.v[jx-1] - self.up[ix  ] ) * self.ap[ix  ] \
                                          + coll2_fac),
                            ((i,  ), j  , + time_fac \
                                          - 2. * coll2_fac),
                            ((i,  ), j+1, - (hh[ix+1, jx  ] - hh[ix-1, jx  ]) * arak_fac_J1 \
                                          - (hh[ix+1, jx+1] - hh[ix-1, jx+1]) * arak_fac_J1 \
                                          + coll1_fac * ( self.v[jx+1] - self.up[ix  ] ) * self.ap[ix  ] \
                                          + coll2_fac),
                            ((i,  ), j+2, - (hh[ix+1, jx+1] - hh[ix-1, jx+1]) * arak_fac_J2),
                            ((i+1,), j-1, + (hh[ix+1, jx  ] - hh[ix,   jx-1]) * arak_fac_J1 \
                                          + (hh[ix+2, jx  ] - hh[ix,   jx-2]) * arak_fac_J2 \
                                          + (hh[ix+1, jx+1] - hh[ix-1, jx-1]) * arak_fac_J2),
                            ((i+1,), j  , + (hh[ix,   jx+1] - hh[ix,   jx-1]) * arak_fac_J1 \
                                          + (hh[ix+1, jx+1] - hh[ix+1, jx-1]) * arak_fac_J1),
                            ((i+1,), j+1, + (hh[ix,   jx+1] - hh[ix+1, jx  ]) * arak_fac_J1 \
                                          + (hh[ix,   jx+2] - hh[ix+2, jx  ]) * arak_fac_J2 \
                                          + (hh[ix-1, jx+1] - hh[ix+1, jx-1]) * arak_fac_J2),
                            ((i+2,), j  , + (hh[ix+1, jx+1] - hh[ix+1, jx-1]) * arak_fac_J2),
                        ]:

                        col.index = index
                        col.field = field
                        A.setValueStencil(row, col, value)
                        
        
        A.assemble()



    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef double jpp_J1, jpc_J1, jcp_J1
        cdef double jcc_J2, jpc_J2, jcp_J2
        cdef double result_J1, result_J2, result_J4, poisson
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double[:,:] fd    = self.da1.getLocalArray(F, self.localFd)
        cdef double[:,:] y     = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[double, ndim=2] h0  = self.da1.getLocalArray(self.H0,  self.localH0)
        cdef npy.ndarray[double, ndim=2] h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
        cdef npy.ndarray[double, ndim=2] h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
        
        cdef double[:,:] hh = h0 + h1h + h2h
        

        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy] = fd[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    # Arakawa's J1
                    jpp_J1 = (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (hh[ix,   jx+1] - hh[ix,   jx-1]) \
                           - (fd[ix,   jx+1] - fd[ix,   jx-1]) * (hh[ix+1, jx  ] - hh[ix-1, jx  ])
                    
                    jpc_J1 = fd[ix+1, jx  ] * (hh[ix+1, jx+1] - hh[ix+1, jx-1]) \
                           - fd[ix-1, jx  ] * (hh[ix-1, jx+1] - hh[ix-1, jx-1]) \
                           - fd[ix,   jx+1] * (hh[ix+1, jx+1] - hh[ix-1, jx+1]) \
                           + fd[ix,   jx-1] * (hh[ix+1, jx-1] - hh[ix-1, jx-1])
                    
                    jcp_J1 = fd[ix+1, jx+1] * (hh[ix,   jx+1] - hh[ix+1, jx  ]) \
                           - fd[ix-1, jx-1] * (hh[ix-1, jx  ] - hh[ix,   jx-1]) \
                           - fd[ix-1, jx+1] * (hh[ix,   jx+1] - hh[ix-1, jx  ]) \
                           + fd[ix+1, jx-1] * (hh[ix+1, jx  ] - hh[ix,   jx-1])
                    
                    # Arakawa's J2
                    jcc_J2 = (fd[ix+1, jx+1] - fd[ix-1, jx-1]) * (hh[ix-1, jx+1] - hh[ix+1, jx-1]) \
                           - (fd[ix-1, jx+1] - fd[ix+1, jx-1]) * (hh[ix+1, jx+1] - hh[ix-1, jx-1])
                    
                    jpc_J2 = fd[ix+2, jx  ] * (hh[ix+1, jx+1] - hh[ix+1, jx-1]) \
                           - fd[ix-2, jx  ] * (hh[ix-1, jx+1] - hh[ix-1, jx-1]) \
                           - fd[ix,   jx+2] * (hh[ix+1, jx+1] - hh[ix-1, jx+1]) \
                           + fd[ix,   jx-2] * (hh[ix+1, jx-1] - hh[ix-1, jx-1])
                    
                    jcp_J2 = fd[ix+1, jx+1] * (hh[ix,   jx+2] - hh[ix+2, jx  ]) \
                           - fd[ix-1, jx-1] * (hh[ix-2, jx  ] - hh[ix,   jx-2]) \
                           - fd[ix-1, jx+1] * (hh[ix,   jx+2] - hh[ix-2, jx  ]) \
                           + fd[ix+1, jx-1] * (hh[ix+2, jx  ] - hh[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    poisson   = 0.5 * result_J4 * self.grid.hx_inv * self.grid.hv_inv
                    
                    # solution
                    y[iy, jy] = fd[ix, jx] * self.grid.ht_inv \
                              + poisson
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec F, Vec Y):
        cdef npy.int64_t i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef double jpp_J1, jpc_J1, jcp_J1
        cdef double jcc_J2, jpc_J2, jcp_J2
        cdef double result_J1, result_J2, result_J4
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double[:,:] y     = self.da1.getGlobalArray(Y)
        cdef double[:,:] fp    = self.da1.getLocalArray(F, self.localFp)
        cdef double[:,:] fh    = self.da1.getLocalArray(self.Fh, self.localFh)
        
        cdef npy.ndarray[double, ndim=2] h0  = self.da1.getLocalArray(self.H0,  self.localH0)
        cdef npy.ndarray[double, ndim=2] h1p = self.da1.getLocalArray(self.H1p, self.localH1p)
        cdef npy.ndarray[double, ndim=2] h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
        cdef npy.ndarray[double, ndim=2] h2p = self.da1.getLocalArray(self.H2p, self.localH2p)
        cdef npy.ndarray[double, ndim=2] h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
        
        cdef double[:,:] hp = h0 + h1p + h2p
        cdef double[:,:] hh = h0 + h1h + h2h
        
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy] = fp[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
                    
                    # time derivative
                    y[iy, jy] = (fp[ix, jx] - fh[ix, jx]) * self.grid.ht_inv
                    
                    
                    # Arakawa's J1 (fp,hh)
                    jpp_J1 = (fp[ix+1, jx  ] - fp[ix-1, jx  ]) * (hh[ix,   jx+1] - hh[ix,   jx-1]) \
                           - (fp[ix,   jx+1] - fp[ix,   jx-1]) * (hh[ix+1, jx  ] - hh[ix-1, jx  ])
                    
                    jpc_J1 = fp[ix+1, jx  ] * (hh[ix+1, jx+1] - hh[ix+1, jx-1]) \
                           - fp[ix-1, jx  ] * (hh[ix-1, jx+1] - hh[ix-1, jx-1]) \
                           - fp[ix,   jx+1] * (hh[ix+1, jx+1] - hh[ix-1, jx+1]) \
                           + fp[ix,   jx-1] * (hh[ix+1, jx-1] - hh[ix-1, jx-1])
                    
                    jcp_J1 = fp[ix+1, jx+1] * (hh[ix,   jx+1] - hh[ix+1, jx  ]) \
                           - fp[ix-1, jx-1] * (hh[ix-1, jx  ] - hh[ix,   jx-1]) \
                           - fp[ix-1, jx+1] * (hh[ix,   jx+1] - hh[ix-1, jx  ]) \
                           + fp[ix+1, jx-1] * (hh[ix+1, jx  ] - hh[ix,   jx-1])
                    
                    # Arakawa's J2 (fp,hh)
                    jcc_J2 = (fp[ix+1, jx+1] - fp[ix-1, jx-1]) * (hh[ix-1, jx+1] - hh[ix+1, jx-1]) \
                           - (fp[ix-1, jx+1] - fp[ix+1, jx-1]) * (hh[ix+1, jx+1] - hh[ix-1, jx-1])
                    
                    jpc_J2 = fp[ix+2, jx  ] * (hh[ix+1, jx+1] - hh[ix+1, jx-1]) \
                           - fp[ix-2, jx  ] * (hh[ix-1, jx+1] - hh[ix-1, jx-1]) \
                           - fp[ix,   jx+2] * (hh[ix+1, jx+1] - hh[ix-1, jx+1]) \
                           + fp[ix,   jx-2] * (hh[ix+1, jx-1] - hh[ix-1, jx-1])
                    
                    jcp_J2 = fp[ix+1, jx+1] * (hh[ix,   jx+2] - hh[ix+2, jx  ]) \
                           - fp[ix-1, jx-1] * (hh[ix-2, jx  ] - hh[ix,   jx-2]) \
                           - fp[ix-1, jx+1] * (hh[ix,   jx+2] - hh[ix-2, jx  ]) \
                           + fp[ix+1, jx-1] * (hh[ix+2, jx  ] - hh[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    # Arakawa's J4 (fp,hh)
                    y[iy, jy] += 0.5 * result_J4 * self.grid.hx_inv * self.grid.hv_inv
                    
                    
                    # Arakawa's J1 (fh,hp)
                    jpp_J1 = (fh[ix+1, jx  ] - fh[ix-1, jx  ]) * (hp[ix,   jx+1] - hp[ix,   jx-1]) \
                           - (fh[ix,   jx+1] - fh[ix,   jx-1]) * (hp[ix+1, jx  ] - hp[ix-1, jx  ])                    
                    
                    jpc_J1 = fh[ix+1, jx  ] * (hp[ix+1, jx+1] - hp[ix+1, jx-1]) \
                           - fh[ix-1, jx  ] * (hp[ix-1, jx+1] - hp[ix-1, jx-1]) \
                           - fh[ix,   jx+1] * (hp[ix+1, jx+1] - hp[ix-1, jx+1]) \
                           + fh[ix,   jx-1] * (hp[ix+1, jx-1] - hp[ix-1, jx-1])
                    
                    jcp_J1 = fh[ix+1, jx+1] * (hp[ix,   jx+1] - hp[ix+1, jx  ]) \
                           - fh[ix-1, jx-1] * (hp[ix-1, jx  ] - hp[ix,   jx-1]) \
                           - fh[ix-1, jx+1] * (hp[ix,   jx+1] - hp[ix-1, jx  ]) \
                           + fh[ix+1, jx-1] * (hp[ix+1, jx  ] - hp[ix,   jx-1])
                    
                    # Arakawa's J2 (fh,hp)
                    jcc_J2 = (fh[ix+1, jx+1] - fh[ix-1, jx-1]) * (hp[ix-1, jx+1] - hp[ix+1, jx-1]) \
                           - (fh[ix-1, jx+1] - fh[ix+1, jx-1]) * (hp[ix+1, jx+1] - hp[ix-1, jx-1])
                    
                    jpc_J2 = fh[ix+2, jx  ] * (hp[ix+1, jx+1] - hp[ix+1, jx-1]) \
                           - fh[ix-2, jx  ] * (hp[ix-1, jx+1] - hp[ix-1, jx-1]) \
                           - fh[ix,   jx+2] * (hp[ix+1, jx+1] - hp[ix-1, jx+1]) \
                           + fh[ix,   jx-2] * (hp[ix+1, jx-1] - hp[ix-1, jx-1])
                    
                    jcp_J2 = fh[ix+1, jx+1] * (hp[ix,   jx+2] - hp[ix+2, jx  ]) \
                           - fh[ix-1, jx-1] * (hp[ix-2, jx  ] - hp[ix,   jx-2]) \
                           - fh[ix-1, jx+1] * (hp[ix,   jx+2] - hp[ix-2, jx  ]) \
                           + fh[ix+1, jx-1] * (hp[ix+2, jx  ] - hp[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    # Arakawa's J4 (fh,hp)
                    y[iy, jy] += 0.5 * result_J4 * self.grid.hx_inv * self.grid.hv_inv
