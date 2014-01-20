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
    Implements a variational integrator with fourth order
    symplectic Runge-Kutta time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self, VIDA da1, VIDA dax, Vec H0,
                 npy.ndarray[npy.float64_t, ndim=1] v,
                 npy.uint64_t nx, npy.uint64_t nv,
                 npy.float64_t ht, npy.float64_t hx, npy.float64_t hv,
                 npy.float64_t charge,
                 npy.float64_t coll_freq=0.,
                 npy.float64_t coll_diff=1.,
                 npy.float64_t coll_drag=1.):
        '''
        Constructor
        '''
        
        # initialise parent class
        super().__init__(da1, dax, H0, v, nx, nv, ht, hx, hv, charge, coll_freq, coll_diff, coll_drag, regularisation=0.)
        
        # k vector
        self.localK = self.da1.createLocalVec()
        
        # create global data arrays
        self.H11 = self.da1.createGlobalVec()
        self.H21 = self.da1.createGlobalVec()
        
        self.F1  = self.da1.createGlobalVec()
        self.P1  = self.dax.createGlobalVec()
        
        # create local data arrays
        self.localH11 = self.da1.createLocalVec()
        self.localH21 = self.da1.createLocalVec()

        self.localF1  = self.da1.createLocalVec()
        self.localP1  = self.dax.createLocalVec()

        
    
    cpdef update_previous2(self, Vec F1, Vec P1, Vec Pext1, Vec N, Vec U, Vec E):
        F1.copy(self.F1)
        P1.copy(self.P1)
        N.copy(self.Np)
        U.copy(self.Up)
        E.copy(self.Ep)
        
        self.toolbox.compute_collision_factor(self.Np,  self.Up, self.Ep, self.Ap)
        self.toolbox.potential_to_hamiltonian(self.P1, self.H11)
        self.toolbox.potential_to_hamiltonian(Pext1,   self.H21)
        
        
    cdef get_data_arrays(self):
        self.h0  = self.da1.getLocalArray(self.H0,  self.localH0 )
        self.h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
        self.h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
        
        self.np  = self.dax.getLocalArray(self.Np,  self.localNp)
        self.up  = self.dax.getLocalArray(self.Up,  self.localUp)
        self.ep  = self.dax.getLocalArray(self.Ep,  self.localEp)
        self.ap  = self.dax.getLocalArray(self.Ap,  self.localAp)
        
        self.fh  = self.da1.getLocalArray(self.Fh,  self.localFh)
        self.ph  = self.dax.getLocalArray(self.Ph,  self.localPh)
        self.nh  = self.dax.getLocalArray(self.Nh,  self.localNh)
        self.uh  = self.dax.getLocalArray(self.Uh,  self.localUh)
        self.eh  = self.dax.getLocalArray(self.Eh,  self.localEh)
        self.ah  = self.dax.getLocalArray(self.Ah,  self.localAh)
        
        self.h11 = self.da1.getLocalArray(self.H11, self.localH11)
        self.h21 = self.da1.getLocalArray(self.H21, self.localH21)
        
        self.f1  = self.da1.getLocalArray(self.F1,  self.localF1)
        self.p1  = self.dax.getLocalArray(self.P1,  self.localP1)

    
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     def formJacobian(self, Mat A):
#         cdef npy.int64_t i, j, ix
#         cdef npy.int64_t xe, xs, ye, ys
#         
#         (xs, xe), (ys, ye) = self.da1.getRanges()
#         
#         self.get_data_arrays()
#         
#         cdef npy.ndarray[npy.float64_t, ndim=2] h_ave = self.h0 + 0.5 * (self.h1h + self.h11p + self.h12p) \
#                                                                 + 0.5 * (self.h2h + self.h2p)
#         
#         
# #         cdef npy.float64_t time_fac      = 0.
# #         cdef npy.float64_t arak_fac_J1   = 0.
# #         cdef npy.float64_t arak_fac_J2   = 0.
# #         cdef npy.float64_t coll_drag_fac = 0.
# #         cdef npy.float64_t coll_diff_fac = 0.
#         
#         cdef npy.float64_t time_fac      = 1.0  / self.grid.ht
#         cdef npy.float64_t arak_fac_J1   = + 1.0 / (12. * self.grid.hx * self.grid.hv)
#         cdef npy.float64_t arak_fac_J2   = - 0.5 / (24. * self.grid.hx * self.grid.hv)
#         
#         cdef npy.float64_t coll_drag_fac = - 0.5 * self.nu * self.coll_drag * self.grid.hv_inv * 0.5
#         cdef npy.float64_t coll_diff_fac = - 0.5 * self.nu * self.coll_diff * self.grid.hv2_inv
#         
#         
#         A.zeroEntries()
#         
#         row = Mat.Stencil()
#         col = Mat.Stencil()
#         
#         
#         # Vlasov Equation
#         for i in range(xs, xe):
#             ix = i-xs+self.da1.getStencilWidth()
#             
#             row.index = (i,)
#                 
#             for j in range(ys, ye):
#                 jx = j-ys+self.da1.getStencilWidth()
#                 jy = j-ys

#                 row.field = j
#                 
#                 # Dirichlet boundary conditions
#                 if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
#                     A.setValueStencil(row, row, 1.0)
#                     
#                 else:
#                     
#                     for index, field, value in [
#                             ((i-2,), j  , - (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
#                             ((i-1,), j-1, - (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) * arak_fac_J1 \
#                                           - (h_ave[ix-2, jx  ] - h_ave[ix,   jx-2]) * arak_fac_J2 \
#                                           - (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J2),
#                             ((i-1,), j  , - (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) * arak_fac_J1 \
#                                           - (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J1),
#                             ((i-1,), j+1, - (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
#                                           - (h_ave[ix,   jx+2] - h_ave[ix-2, jx  ]) * arak_fac_J2 \
#                                           - (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
#                             ((i,  ), j-2, + (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
#                             ((i,  ), j-1, + (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
#                                           + (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) * arak_fac_J1 \
#                                           - coll_drag_fac * ( self.v[jx-1] - self.up[ix  ] ) * self.ap[ix  ] \
#                                           + coll_diff_fac),
#                             ((i,  ), j  , + time_fac \
#                                           - 2. * coll_diff_fac),
#                             ((i,  ), j+1, - (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
#                                           - (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) * arak_fac_J1 \
#                                           + coll_drag_fac * ( self.v[jx+1] - self.up[ix  ] ) * self.ap[ix  ] \
#                                           + coll_diff_fac,
#                             ((i,  ), j+2, - (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) * arak_fac_J2),
#                             ((i+1,), j-1, + (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1]) * arak_fac_J1 \
#                                           + (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2]) * arak_fac_J2 \
#                                           + (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
#                             ((i+1,), j  , + (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) * arak_fac_J1 \
#                                           + (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J1),
#                             ((i+1,), j+1, + (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) * arak_fac_J1 \
#                                           + (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) * arak_fac_J2 \
#                                           + (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J2),
#                             ((i+2,), j  , + (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J2),
#                         ]:
# 
#                         col.index = index
#                         col.field = field
#                         A.setValueStencil(row, col, value)
#                         
#         
#         A.assemble()



    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(self, Vec K, Vec Y):
        cdef npy.int64_t a, i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        cdef npy.float64_t coll_drag, coll_diff
        
        self.get_data_arrays()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] k = self.da1.getLocalArray(K, self.localK)
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] f = npy.empty_like(k)
        cdef npy.ndarray[npy.float64_t, ndim=2] h = npy.empty_like(k)
         
        f[:,:] = 0.5 * self.grid.ht * k[:,:]
        
        h[:,:] = self.h0 + self.h11 + self.h21

#         cdef npy.ndarray[npy.float64_t, ndim=1] v = self.v
#         cdef npy.ndarray[npy.float64_t, ndim=1] u = self.up
#         cdef npy.ndarray[npy.float64_t, ndim=1] a = self.ap
        
        
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
        
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = k[ix, jx]
                    
                else:
                    # Araaawa's J1
                    jpp_J1 = (f[ix+1, jx  ] - f[ix-1, jx  ]) * (h[ix,   jx+1] - h[ix,   jx-1]) \
                           - (f[ix,   jx+1] - f[ix,   jx-1]) * (h[ix+1, jx  ] - h[ix-1, jx  ])
                    
                    jpc_J1 = f[ix+1, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - f[ix-1, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - f[ix,   jx+1] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + f[ix,   jx-1] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J1 = f[ix+1, jx+1] * (h[ix,   jx+1] - h[ix+1, jx  ]) \
                           - f[ix-1, jx-1] * (h[ix-1, jx  ] - h[ix,   jx-1]) \
                           - f[ix-1, jx+1] * (h[ix,   jx+1] - h[ix-1, jx  ]) \
                           + f[ix+1, jx-1] * (h[ix+1, jx  ] - h[ix,   jx-1])
                    
                    # Araaawa's J2
                    jcc_J2 = (f[ix+1, jx+1] - f[ix-1, jx-1]) * (h[ix-1, jx+1] - h[ix+1, jx-1]) \
                           - (f[ix-1, jx+1] - f[ix+1, jx-1]) * (h[ix+1, jx+1] - h[ix-1, jx-1])
                    
                    jpc_J2 = f[ix+2, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - f[ix-2, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - f[ix,   jx+2] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + f[ix,   jx-2] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J2 = f[ix+1, jx+1] * (h[ix,   jx+2] - h[ix+2, jx  ]) \
                           - f[ix-1, jx-1] * (h[ix-2, jx  ] - h[ix,   jx-2]) \
                           - f[ix-1, jx+1] * (h[ix,   jx+2] - h[ix-2, jx  ]) \
                           + f[ix+1, jx-1] * (h[ix+2, jx  ] - h[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12. * self.grid.hx_inv * self.grid.hv_inv
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24. * self.grid.hx_inv * self.grid.hv_inv
                    result_J4 = 2. * result_J1 - result_J2
                    
                    
                    # collision operator
#                         coll_drag = ( (v[jx+1] - u[ix]) * fd[ix, jx+1] - (v[jx-1] - u[ix]) * fd[ix, jx-1] ) * a[ix]
#                         coll_diff = ( fd[ix, jx+1] - 2. * fd[ix, jx] + fd[ix, jx-1] )
                    coll_drag = 0.0
                    coll_diff = 0.0
         
                    y[iy, jy] = k[ix, jx] + result_J4 # \
#                                     + 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
#                                     + 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv

    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec K, Vec Y):
        cdef npy.int64_t a, i, j
        cdef npy.int64_t ix, iy, jx, jy
        cdef npy.int64_t xe, xs, ye, ys
        
        cdef npy.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef npy.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef npy.float64_t result_J1, result_J2, result_J4
        cdef npy.float64_t coll_drag, coll_diff
        
        self.get_data_arrays()
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef npy.ndarray[npy.float64_t, ndim=2] k = self.da1.getLocalArray(K, self.localK)
        cdef npy.ndarray[npy.float64_t, ndim=2] y = self.da1.getGlobalArray(Y)
        
        cdef npy.ndarray[npy.float64_t, ndim=2] f = npy.empty_like(k)
        cdef npy.ndarray[npy.float64_t, ndim=2] h = npy.empty_like(k)
         
        f[:,:] = self.fh[:,:] + 0.5 * self.grid.ht * k[:,:]
        h[:,:] = self.h0 + self.h11 + self.h21
        
#         cdef npy.ndarray[npy.float64_t, ndim=1] v  = self.v
#         cdef npy.ndarray[npy.float64_t, ndim=1] up = self.up
#         cdef npy.ndarray[npy.float64_t, ndim=1] ap = self.ap
#         cdef npy.ndarray[npy.float64_t, ndim=1] uh = self.uh
#         cdef npy.ndarray[npy.float64_t, ndim=1] ah = self.ah
        
        
        for i in range(xs, xe):
            ix = i-xs+self.da1.getStencilWidth()
            iy = i-xs
            
            # Vlasov equation
            for j in range(ys, ye):
                jx = j-ys+self.da1.getStencilWidth()
                jy = j-ys

                if j < self.da1.getStencilWidth() or j >= self.grid.nv-self.da1.getStencilWidth():
                    # Dirichlet Boundary Conditions
                    y[iy, jy] = k[ix, jx]
                    
                else:
                    # Araaawa's J1
                    jpp_J1 = (f[ix+1, jx  ] - f[ix-1, jx  ]) * (h[ix,   jx+1] - h[ix,   jx-1]) \
                           - (f[ix,   jx+1] - f[ix,   jx-1]) * (h[ix+1, jx  ] - h[ix-1, jx  ])
                    
                    jpc_J1 = f[ix+1, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - f[ix-1, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - f[ix,   jx+1] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + f[ix,   jx-1] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J1 = f[ix+1, jx+1] * (h[ix,   jx+1] - h[ix+1, jx  ]) \
                           - f[ix-1, jx-1] * (h[ix-1, jx  ] - h[ix,   jx-1]) \
                           - f[ix-1, jx+1] * (h[ix,   jx+1] - h[ix-1, jx  ]) \
                           + f[ix+1, jx-1] * (h[ix+1, jx  ] - h[ix,   jx-1])
                    
                    # Araaawa's J2
                    jcc_J2 = (f[ix+1, jx+1] - f[ix-1, jx-1]) * (h[ix-1, jx+1] - h[ix+1, jx-1]) \
                           - (f[ix-1, jx+1] - f[ix+1, jx-1]) * (h[ix+1, jx+1] - h[ix-1, jx-1])
                    
                    jpc_J2 = f[ix+2, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - f[ix-2, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - f[ix,   jx+2] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + f[ix,   jx-2] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J2 = f[ix+1, jx+1] * (h[ix,   jx+2] - h[ix+2, jx  ]) \
                           - f[ix-1, jx-1] * (h[ix-2, jx  ] - h[ix,   jx-2]) \
                           - f[ix-1, jx+1] * (h[ix,   jx+2] - h[ix-2, jx  ]) \
                           + f[ix+1, jx-1] * (h[ix+2, jx  ] - h[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12. * self.grid.hx_inv * self.grid.hv_inv
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24. * self.grid.hx_inv * self.grid.hv_inv
                    result_J4 = 2. * result_J1 - result_J2
                    
                    
                    # collision operator
#                         coll_drag = ( (v[jx+1] - up[ix]) * fp[ix, jx+1] - (v[jx-1] - up[ix]) * fp[ix, jx-1] ) * ap[ix] \
#                                   + ( (v[jx+1] - uh[ix]) * fh[ix, jx+1] - (v[jx-1] - uh[ix]) * fh[ix, jx-1] ) * ah[ix]
#                         coll_diff = ( fp[ix, jx+1] - 2. * fp[ix, jx] + fp[ix, jx-1] ) \
#                                   + ( fh[ix, jx+1] - 2. * fh[ix, jx] + fh[ix, jx-1] )
                    coll_drag = 0.0
                    coll_diff = 0.0
                    
                    
                    y[iy, jy] = k[ix, jx] + result_J4 # \
#                                     + 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
#                                     + 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv
