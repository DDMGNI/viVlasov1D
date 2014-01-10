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
        
        (xs, xe), = self.dax.getRanges()
 
        for i in range(0, xe-xs):
            for j in range(0, self.nv):
                h_arr[i,j] = 0.5 * v[j]**2 * mass
 
 
    def initialise_distribution_function(self, np.ndarray[np.float64_t, ndim=2] f_arr,
                                               np.ndarray[np.float64_t, ndim=1] xGrid,
                                               init_function):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe
         
        cdef np.ndarray[np.float64_t, ndim=1] vGrid = self.v
        
        (xs, xe), = self.dax.getRanges()
 
        for i in range(0, xe-xs):
            for j in range(0, self.nv):
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
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
         
        (xs, xe), = self.dax.getRanges()
 
        for i in range(0, xe-xs):
            for j in range(0, self.nv):
                if j < self.da1.getStencilWidth() or j >= self.nv-self.da1.getStencilWidth():
                    f_arr[i,j] = 0.0
                else:
                    f_arr[i,j] = n_arr[i] * fac * exp( - 0.5 * v[j]**2 / T_arr[i] ) 


#     cdef maxwellian(self, np.float64_t temperature, np.float64_t velocity, np.float64_t vOffset):
#         return self.boltzmannian(temperature, 0.5 * pow(velocity+vOffset, 2))
#     
#     
#     cdef boltzmannian(self, np.float64_t temperature, np.float64_t energy):
#         return sqrt(0.5 / np.pi) * exp( - energy / temperature )


