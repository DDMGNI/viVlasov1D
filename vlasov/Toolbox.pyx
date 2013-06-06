'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np


cdef class Toolbox(object):
    '''
    
    '''
    
    def __cinit__(self, VIDA da1, VIDA da2, VIDA dax, 
                  np.ndarray[np.float64_t, ndim=1] v,
                  np.uint64_t  nx, np.uint64_t  nv,
                  np.float64_t ht, np.float64_t hx, np.float64_t hv):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.dax = dax
        self.da1 = da1
        self.da2 = da2
        
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
    cdef arakawa_J1_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in np.arange(xs, xe):
            for j in np.arange(0, self.nv):
                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = - self.arakawa_J1(x, h, ix, j)
                    
    
    
    @cython.boundscheck(False)
    cdef arakawa_J2_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in np.arange(xs, xe):
            for j in np.arange(0, self.nv):
                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = - self.arakawa_J2(x, h, ix, j)
                    
    
    
    @cython.boundscheck(False)
    cdef arakawa_J4_timestep(self, np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=2] y,
                                   np.ndarray[np.float64_t, ndim=2] h0,
                                   np.ndarray[np.float64_t, ndim=2] h1):
        
        cdef np.uint64_t ix, iy, i, j
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=2] h = h0 + h1
        
        (xs, xe), = self.da1.getRanges()
        
        
        for i in np.arange(xs, xe):
            for j in np.arange(0, self.nv):
                ix = i-xs+2
                iy = i-xs
                
                if j <= 1 or j >= self.nv-2:
                    # Dirichlet boundary conditions
                    y[iy, j] = 0.0
                    
                else:
                    # Vlasov equation
                    y[iy, j] = - self.arakawa_J4(x, h, ix, j)
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] f,
                                            np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        return f[i,j] * self.ht_inv
    
    
    
    @cython.boundscheck(False)
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



    def potential_to_hamiltonian(self, Vec P, Vec H):
        cdef np.float64_t phisum, phiave
        
        (xs, xe), = self.dax.getRanges()
        
        p = self.dax.getVecArray(P)
        h = self.da1.getVecArray(H)
        
        phisum = P.sum()
        phiave = phisum / self.nx
        
        for j in np.arange(0, self.nv):
            h[xs:xe, j] = p[xs:xe] - phiave


    def compute_density(self, Vec F, Vec N):
        f = self.da1.getGlobalArray(F)
        n = self.dax.getGlobalArray(N)
        
        self.compute_density_array(f, n)
    
    
    def compute_velocity_density(self, Vec F, Vec U):
        f = self.da1.getGlobalArray(F)
        u = self.dax.getGlobalArray(U)
        
        self.compute_velocity_density_array(f, u)
    
    
    def compute_energy_density(self, Vec F, Vec E):
        f = self.da1.getGlobalArray(F)
        e = self.dax.getGlobalArray(E)
        
        self.compute_energy_density_array(f, e)
    
    
    def compute_collision_factor(self, Vec N, Vec U, Vec E, Vec A):
        n = self.dax.getGlobalArray(N)
        u = self.dax.getGlobalArray(U)
        e = self.dax.getGlobalArray(E)
        a = self.dax.getGlobalArray(A)
        
        a[:] = n**2 / (n*e - u**2)
    
    
    cdef compute_density_array(self, np.ndarray[np.float64_t, ndim=2] f, np.ndarray[np.float64_t, ndim=1] n):
        (xs, xe), = self.dax.getRanges()
        
        for i in np.arange(xs, xe):
            ix = i-xs
            iy = i-xs
            
            n[iy] = 0.
            
            for j in np.arange(0, (self.nv-1)/2):
                n[iy] += f[ix, j] + f[ix, self.nv-1-j]

            n[iy] += f[ix, (self.nv-1)/2]
                
            n[iy] *= self.hv
    

    cdef compute_velocity_density_array(self, np.ndarray[np.float64_t, ndim=2] f, np.ndarray[np.float64_t, ndim=1] u):
        (xs, xe), = self.dax.getRanges()
        
        for i in np.arange(xs, xe):
            ix = i-xs
            iy = i-xs
            
            u[iy] = 0.
            
            for j in np.arange(0, (self.nv-1)/2):
                u[iy] += self.v[j] * f[ix, j] + self.v[self.nv-1-j] * f[ix, self.nv-1-j]

            u[iy] += self.v[(self.nv-1)/2] * f[ix, (self.nv-1)/2]
                
            u[iy] *= self.hv
    

    cdef compute_energy_density_array(self, np.ndarray[np.float64_t, ndim=2] f, np.ndarray[np.float64_t, ndim=1] e):
        (xs, xe), = self.dax.getRanges()
        
        for i in np.arange(xs, xe):
            ix = i-xs
            iy = i-xs
            
            e[iy] = 0.
            
            for j in np.arange(0, (self.nv-1)/2):
                e[iy] += self.v[j]**2 * f[ix, j] + self.v[self.nv-1-j]**2 * f[ix, self.nv-1-j]

            e[iy] += self.v[(self.nv-1)/2]**2 * f[ix, (self.nv-1)/2]
                
            e[iy] *= self.hv
    

