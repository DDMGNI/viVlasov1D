'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from libc.math cimport exp, pow, sqrt


cdef class Toolbox(object):
    '''
    
    '''
    
    def __cinit__(self, VIDA da1, VIDA dax, 
                  np.ndarray[np.float64_t, ndim=1] v,
                  np.uint64_t  nx, np.uint64_t  nv,
                  np.float64_t ht, np.float64_t hx, np.float64_t hv):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.dax = dax
        self.da1 = da1
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv

        self.ht_inv  = 1. / self.ht 
        self.hx_inv  = 1. / self.hx 
        self.hv_inv  = 1. / self.hv
         
        self.hx2     = hx**2
        self.hx2_inv = 1. / self.hx2 
        
        self.hv2     = hv**2
        self.hv2_inv = 1. / self.hv2 
        
        # velocity grid
        self.v = v.copy()
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t arakawa_J1(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.ndarray[np.float64_t, ndim=2] h,
                                       np.uint64_t i, np.uint64_t j):
        '''
        Arakawa Bracket
        '''
        
        cdef np.float64_t jpp, jpc, jcp, result
        
        jpp = (f[i+1, j  ] - f[i-1, j  ]) * (h[i,   j+1] - h[i,   j-1]) \
            - (f[i,   j+1] - f[i,   j-1]) * (h[i+1, j  ] - h[i-1, j  ])
        
        jpc = f[i+1, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
            - f[i-1, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
            - f[i,   j+1] * (h[i+1, j+1] - h[i-1, j+1]) \
            + f[i,   j-1] * (h[i+1, j-1] - h[i-1, j-1])
        
        jcp = f[i+1, j+1] * (h[i,   j+1] - h[i+1, j  ]) \
            - f[i-1, j-1] * (h[i-1, j  ] - h[i,   j-1]) \
            - f[i-1, j+1] * (h[i,   j+1] - h[i-1, j  ]) \
            + f[i+1, j-1] * (h[i+1, j  ] - h[i,   j-1])
        
        result = (jpp + jpc + jcp) / (12. * self.hx * self.hv)
        
        return result
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t arakawa_J2(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.ndarray[np.float64_t, ndim=2] h,
                                       np.uint64_t i, np.uint64_t j):
        '''
        Arakawa Bracket
        '''
        
        cdef np.float64_t jcc, jpc, jcp, result
        
        jcc = (f[i+1, j+1] - f[i-1, j-1]) * (h[i-1, j+1] - h[i+1, j-1]) \
            - (f[i-1, j+1] - f[i+1, j-1]) * (h[i+1, j+1] - h[i-1, j-1])
        
        jpc = f[i+2, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
            - f[i-2, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
            - f[i,   j+2] * (h[i+1, j+1] - h[i-1, j+1]) \
            + f[i,   j-2] * (h[i+1, j-1] - h[i-1, j-1])
        
        jcp = f[i+1, j+1] * (h[i,   j+2] - h[i+2, j  ]) \
            - f[i-1, j-1] * (h[i-2, j  ] - h[i,   j-2]) \
            - f[i-1, j+1] * (h[i,   j+2] - h[i-2, j  ]) \
            + f[i+1, j-1] * (h[i+2, j  ] - h[i,   j-2])
        
        result = (jcc + jpc + jcp) / (24. * self.hx * self.hv)
        
        return result
    
    
    
    cdef np.float64_t arakawa_J4(self, np.ndarray[np.float64_t, ndim=2] f,
                                       np.ndarray[np.float64_t, ndim=2] h,
                                       np.uint64_t i, np.uint64_t j):
        '''
        Arakawa Bracket 4th order
        '''
        
        return 2.0 * self.arakawa_J1(f, h, i, j) - self.arakawa_J2(f, h, i, j)
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef arakawa_J1_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            for j in range(0, self.nv):
                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = - self.arakawa_J1(x, h, ix, j)
                    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef arakawa_J2_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            for j in range(0, self.nv):
                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = - self.arakawa_J2(x, h, ix, j)
                    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef arakawa_J4_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef np.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef np.float64_t result_J1, result_J2, result_J4
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            for j in range(0, self.nv):
                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
#                     y[iy, j] = - self.arakawa_J4(x, h, ix, j)
    
                    jpp_J1 = (x[ix+1, j  ] - x[ix-1, j  ]) * (h[ix,   j+1] - h[ix,   j-1]) \
                           - (x[ix,   j+1] - x[ix,   j-1]) * (h[ix+1, j  ] - h[ix-1, j  ])
                    
                    jpc_J1 = x[ix+1, j  ] * (h[ix+1, j+1] - h[ix+1, j-1]) \
                           - x[ix-1, j  ] * (h[ix-1, j+1] - h[ix-1, j-1]) \
                           - x[ix,   j+1] * (h[ix+1, j+1] - h[ix-1, j+1]) \
                           + x[ix,   j-1] * (h[ix+1, j-1] - h[ix-1, j-1])
                    
                    jcp_J1 = x[ix+1, j+1] * (h[ix,   j+1] - h[ix+1, j  ]) \
                           - x[ix-1, j-1] * (h[ix-1, j  ] - h[ix,   j-1]) \
                           - x[ix-1, j+1] * (h[ix,   j+1] - h[ix-1, j  ]) \
                           + x[ix+1, j-1] * (h[ix+1, j  ] - h[ix,   j-1])
                    
                    jcc_J2 = (x[ix+1, j+1] - x[ix-1, j-1]) * (h[ix-1, j+1] - h[ix+1, j-1]) \
                           - (x[ix-1, j+1] - x[ix+1, j-1]) * (h[ix+1, j+1] - h[ix-1, j-1])
                    
                    jpc_J2 = x[ix+2, j  ] * (h[ix+1, j+1] - h[ix+1, j-1]) \
                           - x[ix-2, j  ] * (h[ix-1, j+1] - h[ix-1, j-1]) \
                           - x[ix,   j+2] * (h[ix+1, j+1] - h[ix-1, j+1]) \
                           + x[ix,   j-2] * (h[ix+1, j-1] - h[ix-1, j-1])
                    
                    jcp_J2 = x[ix+1, j+1] * (h[ix,   j+2] - h[ix+2, j  ]) \
                           - x[ix-1, j-1] * (h[ix-2, j  ] - h[ix,   j-2]) \
                           - x[ix-1, j+1] * (h[ix,   j+2] - h[ix-2, j  ]) \
                           + x[ix+1, j-1] * (h[ix+2, j  ] - h[ix,   j-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    
                    y[iy, j] = - result_J4 * self.hx_inv * self.hv_inv
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef arakawa_J4_timestep_h(self, np.ndarray[np.float64_t, ndim=2] x,
                                     np.ndarray[np.float64_t, ndim=2] y,
                                     np.ndarray[np.float64_t, ndim=2] h):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.float64_t jpp_J1, jpc_J1, jcp_J1
        cdef np.float64_t jcc_J2, jpc_J2, jcp_J2
        cdef np.float64_t result_J1, result_J2, result_J4
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in range(xs, xe):
            for j in range(0, self.nv):
                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
    
                    jpp_J1 = (x[ix+1, j  ] - x[ix-1, j  ]) * (h[ix,   j+1] - h[ix,   j-1]) \
                           - (x[ix,   j+1] - x[ix,   j-1]) * (h[ix+1, j  ] - h[ix-1, j  ])
                    
                    jpc_J1 = x[ix+1, j  ] * (h[ix+1, j+1] - h[ix+1, j-1]) \
                           - x[ix-1, j  ] * (h[ix-1, j+1] - h[ix-1, j-1]) \
                           - x[ix,   j+1] * (h[ix+1, j+1] - h[ix-1, j+1]) \
                           + x[ix,   j-1] * (h[ix+1, j-1] - h[ix-1, j-1])
                    
                    jcp_J1 = x[ix+1, j+1] * (h[ix,   j+1] - h[ix+1, j  ]) \
                           - x[ix-1, j-1] * (h[ix-1, j  ] - h[ix,   j-1]) \
                           - x[ix-1, j+1] * (h[ix,   j+1] - h[ix-1, j  ]) \
                           + x[ix+1, j-1] * (h[ix+1, j  ] - h[ix,   j-1])
                    
                    jcc_J2 = (x[ix+1, j+1] - x[ix-1, j-1]) * (h[ix-1, j+1] - h[ix+1, j-1]) \
                           - (x[ix-1, j+1] - x[ix+1, j-1]) * (h[ix+1, j+1] - h[ix-1, j-1])
                    
                    jpc_J2 = x[ix+2, j  ] * (h[ix+1, j+1] - h[ix+1, j-1]) \
                           - x[ix-2, j  ] * (h[ix-1, j+1] - h[ix-1, j-1]) \
                           - x[ix,   j+2] * (h[ix+1, j+1] - h[ix-1, j+1]) \
                           + x[ix,   j-2] * (h[ix+1, j-1] - h[ix-1, j-1])
                    
                    jcp_J2 = x[ix+1, j+1] * (h[ix,   j+2] - h[ix+2, j  ]) \
                           - x[ix-1, j-1] * (h[ix-2, j  ] - h[ix,   j-2]) \
                           - x[ix-1, j+1] * (h[ix,   j+2] - h[ix-2, j  ]) \
                           + x[ix+1, j-1] * (h[ix+2, j  ] - h[ix,   j-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) / 12.
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) / 24.
                    result_J4 = 2. * result_J1 - result_J2
                    
                    
                    y[iy, j] = - result_J4 * self.hx_inv * self.hv_inv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] f,
                                            np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        return f[i,j] * self.ht_inv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t collT1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        return ( (self.v[j+1] - U[i  ]) * f[i,   j+1] - (self.v[j-1] - U[i  ]) * f[i,   j-1] ) * A[i  ] * 0.5 / self.hv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t collT2(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        return ( f[i,   j+1] - 2. * f[i,   j  ] + f[i,   j-1] ) * self.hv2_inv



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t collE1(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        return ( (N[i  ] * self.v[j+1] - U[i  ]) * f[i,   j+1] - (N[i  ] * self.v[j-1] - U[i  ]) * f[i,   j-1] ) * A[i  ] * 0.5 / self.hv
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t collE2(self, np.ndarray[np.float64_t, ndim=2] f,
                                   np.ndarray[np.float64_t, ndim=1] N,
                                   np.ndarray[np.float64_t, ndim=1] U,
                                   np.ndarray[np.float64_t, ndim=1] E,
                                   np.ndarray[np.float64_t, ndim=1] A,
                                   np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        return ( f[i,   j+1] - 2. * f[i,   j  ] + f[i,   j-1] ) * self.hv2_inv



    cpdef potential_to_hamiltonian(self, Vec P, Vec H):
        cdef np.float64_t phisum, phiave
        
        (xs, xe), = self.dax.getRanges()
        
        p = self.dax.getVecArray(P)
        h = self.da1.getVecArray(H)
        
        phisum = P.sum()
        phiave = phisum / self.nx
        
        for j in range(0, self.nv):
            h[xs:xe, j] = p[xs:xe] - phiave


    cpdef compute_density(self, Vec F, Vec N):
        f = self.da1.getGlobalArray(F)
        n = self.dax.getGlobalArray(N)
        
        self.compute_density_array(f, n)
    
    
    cpdef compute_velocity_density(self, Vec F, Vec U):
        f = self.da1.getGlobalArray(F)
        u = self.dax.getGlobalArray(U)
        
        self.compute_velocity_density_array(f, u)
    
    
    cpdef compute_energy_density(self, Vec F, Vec E):
        f = self.da1.getGlobalArray(F)
        e = self.dax.getGlobalArray(E)
        
        self.compute_energy_density_array(f, e)
    
    
    cpdef compute_collision_factor(self, Vec N, Vec U, Vec E, Vec A):
        n = self.dax.getGlobalArray(N)
        u = self.dax.getGlobalArray(U)
        e = self.dax.getGlobalArray(E)
        a = self.dax.getGlobalArray(A)
        
        self.compute_collision_factor_array(n, u, e, a)
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef compute_density_array(self, np.ndarray[np.float64_t, ndim=2] f, np.ndarray[np.float64_t, ndim=1] n):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe
        
        (xs, xe), = self.dax.getRanges()
        
        for i in range(0, xe-xs):
            n[i] = 0.
             
#             for j in range(0, (self.nv-1)/2):
#                 n[i] += f[i,j] + f[i, self.nv-1-j]
# 
#             n[i] += f[i, (self.nv-1)/2]
        
            for j in range(0, self.nv):
                n[i] += f[i,j]
                
            n[i] *= self.hv
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef compute_velocity_density_array(self, np.ndarray[np.float64_t, ndim=2] f, np.ndarray[np.float64_t, ndim=1] u):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
        
        (xs, xe), = self.dax.getRanges()
        
        for i in range(0, xe-xs):
            u[i] = 0.
            
#             for j in range(0, (self.nv-1)/2):
#                 u[i] += v[j] * f[i,j] + v[self.nv-1-j] * f[i, self.nv-1-j]
# 
#             u[i] += v[(self.nv-1)/2] * f[i, (self.nv-1)/2]
                
            for j in range(0, self.nv):
                u[i] += v[j] * f[i,j]
            
            u[i] *= self.hv
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef compute_energy_density_array(self, np.ndarray[np.float64_t, ndim=2] f, np.ndarray[np.float64_t, ndim=1] e):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
        
        (xs, xe), = self.dax.getRanges()
        
        for i in range(0, xe-xs):
            e[i] = 0.
            
#             for j in range(0, (self.nv-1)/2):
#                 e[i] += v[j]**2 * f[i,j] + v[self.nv-1-j]**2 * f[i, self.nv-1-j]
# 
#             e[i] += v[(self.nv-1)/2]**2 * f[i, (self.nv-1)/2]
                
            for j in range(0, self.nv):
                e[i] += v[j]**2 * f[i,j]
            
            e[i] *= self.hv
    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef compute_collision_factor_array(self, np.ndarray[np.float64_t, ndim=1] n,
                                              np.ndarray[np.float64_t, ndim=1] u,
                                              np.ndarray[np.float64_t, ndim=1] e,
                                              np.ndarray[np.float64_t, ndim=1] a):
        cdef np.uint64_t i
        cdef np.uint64_t xs, xe
        
        (xs, xe), = self.dax.getRanges()
        
        for i in range(0, xe-xs):
            a[i] = n[i]**2 / (n[i] * e[i] - u[i]**2)


    def initialise_kinetic_hamiltonian(self, Vec H, np.float64_t mass):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe
         
        cdef np.ndarray[np.float64_t, ndim=2] h_arr = self.da1.getGlobalArray(H)
        cdef np.ndarray[np.float64_t, ndim=1] v     = self.v
        
        (xs, xe), = self.da1.getRanges()
 
        for i in range(0, xe-xs):
            for j in range(0, self.nv):
                h_arr[i,j] = 0.5 * v[j]**2 * mass
 
 
    def initialise_distribution_function(self, np.ndarray[np.float64_t, ndim=2] f_arr,
                                               np.ndarray[np.float64_t, ndim=1] xGrid,
                                               init_function):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe
         
        cdef np.ndarray[np.float64_t, ndim=1] vGrid = self.v
        
        (xs, xe), = self.da1.getRanges()
 
        for i in range(0, xe-xs):
            for j in range(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    f_arr[i,j] = 0.0
                else:
                    f_arr[i,j] = init_function(xGrid[i], vGrid[j]) 
 
 
    def initialise_distribution_nT(self, np.ndarray[np.float64_t, ndim=2] f_arr,
                                         np.ndarray[np.float64_t, ndim=1] n_arr,
                                         np.ndarray[np.float64_t, ndim=1] T_arr):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe
         
        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
         
        cdef np.float64_t pi  = np.pi
        cdef np.float64_t fac = sqrt(0.5 / pi)
         
        (xs, xe), = self.da1.getRanges()
 
        for i in range(0, xe-xs):
            for j in range(0, self.nv):
                if j <= 1 or j >= self.nv-2:
                    f_arr[i,j] = 0.0
                else:
                    f_arr[i,j] = n_arr[i] * fac * exp( - 0.5 * v[j]**2 / T_arr[i] ) 


#     cdef maxwellian(self, np.float64_t temperature, np.float64_t velocity, np.float64_t vOffset):
#         return self.boltzmannian(temperature, 0.5 * pow(velocity+vOffset, 2))
#     
#     
#     cdef boltzmannian(self, np.float64_t temperature, np.float64_t energy):
#         return sqrt(0.5 / np.pi) * exp( - energy / temperature )


