'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from vlasov.toolbox.Toolbox import Toolbox


cdef class PETScVlasovSolver(PETScVlasovSolverBase):
    '''
    Implements a variational integrator with fourth order
    symplectic Runge-Kutta time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 object da2  not None,
                 object da1  not None,
                 Grid grid not None,
                 Vec H0  not None,
                 Vec H1p not None,
                 Vec H1h not None,
                 Vec H2p not None,
                 Vec H2h not None,
                 Vec H11 not None,
                 Vec H12 not None,
                 Vec H21 not None,
                 Vec H22 not None,
                 double charge=-1.,
                 double coll_freq=0.,
                 double coll_diff=1.,
                 double coll_drag=1.,
                 double regularisation=0.):
        '''
        Constructor
        '''
        
        # initialise parent class
        super().__init__(da1, grid, H0, H1p, H1h, H2p, H2h, charge, coll_freq, coll_diff, coll_drag, regularisation)
        
        # distributed array
        self.da2 = da2
        
        # Hamiltonians
        self.H11  = H11
        self.H12  = H12
        self.H21  = H21
        self.H22  = H22
        
        # create local data arrays
        self.localK    = self.da2.createLocalVec()

        self.localH0   = self.da1.createLocalVec()
        self.localH11  = self.da1.createLocalVec()
        self.localH12  = self.da1.createLocalVec()
        self.localH21  = self.da1.createLocalVec()
        self.localH22  = self.da1.createLocalVec()
        
        # create temporary array
        K = self.da2.createGlobalVec()
        self.f_arr = np.empty_like(self.da2.getLocalArray(K, self.localK))
        self.h_arr = np.empty_like(self.da2.getLocalArray(K, self.localK))
        K.destroy()
        
        self.f = self.f_arr
        self.h = self.h_arr
        
        # Runge-Kutta factors
        self.a11 = 0.25
        self.a12 = 0.25 - 0.5 / np.sqrt(3.) 
        self.a21 = 0.25 + 0.5 / np.sqrt(3.) 
        self.a22 = 0.25
        
    
    cpdef update_previous4(self):
        cdef np.ndarray[double, ndim=2] h0  = getLocalArray(self.da1, self.H0,  self.localH0)
        cdef np.ndarray[double, ndim=2] h11 = getLocalArray(self.da1, self.H11, self.localH11)
        cdef np.ndarray[double, ndim=2] h12 = getLocalArray(self.da1, self.H12, self.localH12)
        cdef np.ndarray[double, ndim=2] h21 = getLocalArray(self.da1, self.H21, self.localH21)
        cdef np.ndarray[double, ndim=2] h22 = getLocalArray(self.da1, self.H22, self.localH22)
        
        self.h_arr[:,:,0] = h0[:,:] + h11[:,:] + h21[:,:]  
        self.h_arr[:,:,1] = h0[:,:] + h12[:,:] + h22[:,:]  
        
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef jacobian(self, Vec K, Vec Y):
        cdef int a, i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        cdef double jpp_J1, jpc_J1, jcp_J1
        cdef double jcc_J2, jpc_J2, jcp_J2
        cdef double result_J1, result_J2, result_J4, poisson
        
        cdef np.ndarray[double, ndim=3] k_arr = self.da2.getLocalArray(K, self.localK)
        
        self.f_arr[:,:,0] = self.grid.ht * self.a11 * k_arr[:,:,0] + self.grid.ht * self.a12 * k_arr[:,:,1]
        self.f_arr[:,:,1] = self.grid.ht * self.a21 * k_arr[:,:,0] + self.grid.ht * self.a22 * k_arr[:,:,1]
        
        cdef double[:,:,:] k = k_arr
        cdef double[:,:,:] y = self.da2.getGlobalArray(Y)
        
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy, 0:2] = k[self.grid.stencil:xe-xs+self.grid.stencil, jx, 0:2]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    for a in range(0,2):
                        # Araaawa's J1
                        jpp_J1 = (self.f[ix+1, jx,   a] - self.f[ix-1, jx,   a]) * (self.h[ix,   jx+1, a] - self.h[ix,   jx-1, a]) \
                               - (self.f[ix,   jx+1, a] - self.f[ix,   jx-1, a]) * (self.h[ix+1, jx,   a] - self.h[ix-1, jx,   a])
                        
                        jpc_J1 = self.f[ix+1, jx,   a] * (self.h[ix+1, jx+1, a] - self.h[ix+1, jx-1, a]) \
                               - self.f[ix-1, jx,   a] * (self.h[ix-1, jx+1, a] - self.h[ix-1, jx-1, a]) \
                               - self.f[ix,   jx+1, a] * (self.h[ix+1, jx+1, a] - self.h[ix-1, jx+1, a]) \
                               + self.f[ix,   jx-1, a] * (self.h[ix+1, jx-1, a] - self.h[ix-1, jx-1, a])
                        
                        jcp_J1 = self.f[ix+1, jx+1, a] * (self.h[ix,   jx+1, a] - self.h[ix+1, jx,   a]) \
                               - self.f[ix-1, jx-1, a] * (self.h[ix-1, jx,   a] - self.h[ix,   jx-1, a]) \
                               - self.f[ix-1, jx+1, a] * (self.h[ix,   jx+1, a] - self.h[ix-1, jx,   a]) \
                               + self.f[ix+1, jx-1, a] * (self.h[ix+1, jx,   a] - self.h[ix,   jx-1, a])
                        
                        # Araaawa's J2
                        jcc_J2 = (self.f[ix+1, jx+1, a] - self.f[ix-1, jx-1, a]) * (self.h[ix-1, jx+1, a] - self.h[ix+1, jx-1, a]) \
                               - (self.f[ix-1, jx+1, a] - self.f[ix+1, jx-1, a]) * (self.h[ix+1, jx+1, a] - self.h[ix-1, jx-1, a])
                        
                        jpc_J2 = self.f[ix+2, jx,   a] * (self.h[ix+1, jx+1, a] - self.h[ix+1, jx-1, a]) \
                               - self.f[ix-2, jx,   a] * (self.h[ix-1, jx+1, a] - self.h[ix-1, jx-1, a]) \
                               - self.f[ix,   jx+2, a] * (self.h[ix+1, jx+1, a] - self.h[ix-1, jx+1, a]) \
                               + self.f[ix,   jx-2, a] * (self.h[ix+1, jx-1, a] - self.h[ix-1, jx-1, a])
                        
                        jcp_J2 = self.f[ix+1, jx+1, a] * (self.h[ix,   jx+2, a] - self.h[ix+2, jx,   a]) \
                               - self.f[ix-1, jx-1, a] * (self.h[ix-2, jx,   a] - self.h[ix,   jx-2, a]) \
                               - self.f[ix-1, jx+1, a] * (self.h[ix,   jx+2, a] - self.h[ix-2, jx,   a]) \
                               + self.f[ix+1, jx-1, a] * (self.h[ix+2, jx,   a] - self.h[ix,   jx-2, a])
                        
                        result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                        result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                        result_J4 = 2. * result_J1 - result_J2
                        poisson   = result_J4 * self.grid.hx_inv * self.grid.hv_inv
                        
                        # solution
                        y[iy, jy, a] = k[ix, jx, a] + poisson
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef function(self, Vec K, Vec Y):
        cdef int a, i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        cdef double jpp_J1, jpc_J1, jcp_J1
        cdef double jcc_J2, jpc_J2, jcp_J2
        cdef double result_J1, result_J2, result_J4, poisson
        
        cdef np.ndarray[double, ndim=3] k_arr = self.da2.getLocalArray(K, self.localK)
        cdef np.ndarray[double, ndim=2] fh    = getLocalArray(self.da1, self.Fh, self.localFh)
        
        self.f_arr[:,:,0] = fh[:,:] + self.grid.ht * self.a11 * k_arr[:,:,0] + self.grid.ht * self.a12 * k_arr[:,:,1]
        self.f_arr[:,:,1] = fh[:,:] + self.grid.ht * self.a21 * k_arr[:,:,0] + self.grid.ht * self.a22 * k_arr[:,:,1]
        
        cdef double[:,:,:] k = k_arr
        cdef double[:,:,:] y = self.da2.getGlobalArray(Y)

        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy, 0:2] = k[self.grid.stencil:xe-xs+self.grid.stencil, jx, 0:2]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    for a in range(0,2):
                        # Araaawa's J1
                        jpp_J1 = (self.f[ix+1, jx,   a] - self.f[ix-1, jx,   a]) * (self.h[ix,   jx+1, a] - self.h[ix,   jx-1, a]) \
                               - (self.f[ix,   jx+1, a] - self.f[ix,   jx-1, a]) * (self.h[ix+1, jx,   a] - self.h[ix-1, jx,   a])
                        
                        jpc_J1 = self.f[ix+1, jx,   a] * (self.h[ix+1, jx+1, a] - self.h[ix+1, jx-1, a]) \
                               - self.f[ix-1, jx,   a] * (self.h[ix-1, jx+1, a] - self.h[ix-1, jx-1, a]) \
                               - self.f[ix,   jx+1, a] * (self.h[ix+1, jx+1, a] - self.h[ix-1, jx+1, a]) \
                               + self.f[ix,   jx-1, a] * (self.h[ix+1, jx-1, a] - self.h[ix-1, jx-1, a])
                        
                        jcp_J1 = self.f[ix+1, jx+1, a] * (self.h[ix,   jx+1, a] - self.h[ix+1, jx,   a]) \
                               - self.f[ix-1, jx-1, a] * (self.h[ix-1, jx,   a] - self.h[ix,   jx-1, a]) \
                               - self.f[ix-1, jx+1, a] * (self.h[ix,   jx+1, a] - self.h[ix-1, jx,   a]) \
                               + self.f[ix+1, jx-1, a] * (self.h[ix+1, jx,   a] - self.h[ix,   jx-1, a])
                        
                        # Araaawa's J2
                        jcc_J2 = (self.f[ix+1, jx+1, a] - self.f[ix-1, jx-1, a]) * (self.h[ix-1, jx+1, a] - self.h[ix+1, jx-1, a]) \
                               - (self.f[ix-1, jx+1, a] - self.f[ix+1, jx-1, a]) * (self.h[ix+1, jx+1, a] - self.h[ix-1, jx-1, a])
                        
                        jpc_J2 = self.f[ix+2, jx,   a] * (self.h[ix+1, jx+1, a] - self.h[ix+1, jx-1, a]) \
                               - self.f[ix-2, jx,   a] * (self.h[ix-1, jx+1, a] - self.h[ix-1, jx-1, a]) \
                               - self.f[ix,   jx+2, a] * (self.h[ix+1, jx+1, a] - self.h[ix-1, jx+1, a]) \
                               + self.f[ix,   jx-2, a] * (self.h[ix+1, jx-1, a] - self.h[ix-1, jx-1, a])
                        
                        jcp_J2 = self.f[ix+1, jx+1, a] * (self.h[ix,   jx+2, a] - self.h[ix+2, jx,   a]) \
                               - self.f[ix-1, jx-1, a] * (self.h[ix-2, jx,   a] - self.h[ix,   jx-2, a]) \
                               - self.f[ix-1, jx+1, a] * (self.h[ix,   jx+2, a] - self.h[ix-2, jx,   a]) \
                               + self.f[ix+1, jx-1, a] * (self.h[ix+2, jx,   a] - self.h[ix,   jx-2, a])
                        
                        result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                        result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                        result_J4 = 2. * result_J1 - result_J2
                        poisson   = result_J4 * self.grid.hx_inv * self.grid.hv_inv
                        
                        # solution
                        y[iy, jy, a] = k[ix, jx, a] + poisson
