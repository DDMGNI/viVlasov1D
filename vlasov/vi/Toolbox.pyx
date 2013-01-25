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
    
    def __cinit__(self, DA da1, DA da2, DA dax, 
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
        
        # create local vectors
        self.localF   = da1.createLocalVec()
    
    
    @cython.boundscheck(False)
    cdef np.float64_t arakawa(self, np.ndarray[np.float64_t, ndim=2] f,
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
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] f,
                                            np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * f[i-1, j-1] \
                   + 2. * f[i-1, j  ] \
                   + 1. * f[i-1, j+1] \
                   + 2. * f[i,   j-1] \
                   + 4. * f[i,   j  ] \
                   + 2. * f[i,   j+1] \
                   + 1. * f[i+1, j-1] \
                   + 2. * f[i+1, j  ] \
                   + 1. * f[i+1, j+1] \
                 ) / (16. * self.ht)
        
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t coll1(self, np.ndarray[np.float64_t, ndim=2] f,
                                  np.ndarray[np.float64_t, ndim=1] A1,
                                  np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        cdef np.ndarray[np.float64_t, ndim=1] v = self.v
        
        cdef np.float64_t result
        
        result = 0.25 * ( \
                          - 1. * ( (A1[i-1] - v[j+1]) * f[i-1, j+1] - (A1[i-1] - v[j-1]) * f[i-1, j-1] ) \
                          - 2. * ( (A1[i  ] - v[j+1]) * f[i,   j+1] - (A1[i  ] - v[j-1]) * f[i,   j-1] ) \
                          - 1. * ( (A1[i+1] - v[j+1]) * f[i+1, j+1] - (A1[i+1] - v[j-1]) * f[i+1, j-1] ) \
                        ) * 0.5 / self.hv
        
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t coll2(self, np.ndarray[np.float64_t, ndim=2] f,
                                  np.ndarray[np.float64_t, ndim=1] A2,
                                  np.uint64_t i, np.uint64_t j):
        '''
        Collision Operator
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * ( f[i-1, j+1] - 2. * f[i-1, j  ] + f[i-1, j-1] ) * A2[i-1] \
                     + 2. * ( f[i,   j+1] - 2. * f[i,   j  ] + f[i,   j-1] ) * A2[i  ] \
                     + 1. * ( f[i+1, j+1] - 2. * f[i+1, j  ] + f[i+1, j-1] ) * A2[i+1] \
                 ) * 0.25 * self.hv2_inv
        
        return result



    @cython.boundscheck(False)
    cdef np.float64_t coll_moments(self, Vec F, Vec A1, Vec A2):
        '''
        Calculate Moments of distribution function
        '''
        
        cdef np.int64_t i, j, ix, iy, xs, xe
        
        (xs, xe), = self.da2.getRanges()
        
        
        self.da1.globalToLocal(F, self.localF)
        
        cdef np.ndarray[np.float64_t, ndim=2] f  = self.da1.getVecArray(self.localF)[...]
        cdef np.ndarray[np.float64_t, ndim=1] a1 = self.dax.getVecArray(A1)[...]
        cdef np.ndarray[np.float64_t, ndim=1] a2 = self.dax.getVecArray(A2)[...]
        
        cdef np.ndarray[np.float64_t, ndim=1] mom_n = np.zeros_like(a1)         # density
        cdef np.ndarray[np.float64_t, ndim=1] mom_u = np.zeros_like(a1)         # mean velocity
        cdef np.ndarray[np.float64_t, ndim=1] mom_e = np.zeros_like(a1)         # energy
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            mom_n[iy] = 0.
            mom_u[iy] = 0.
            mom_e[iy] = 0.
            
            for j in np.arange(0, (self.nv-1)/2):
                mom_n[iy] += f[ix, j] + f[ix, self.nv-1-j]
                mom_u[iy] += self.v[j]    * f[ix, j] + self.v[self.nv-1-j]    * f[ix, self.nv-1-j]
                mom_e[iy] += self.v[j]**2 * f[ix, j] + self.v[self.nv-1-j]**2 * f[ix, self.nv-1-j]

            mom_n[iy] += f[ix, (self.nv-1)/2]
            mom_u[iy] += self.v[(self.nv-1)/2]    * f[ix, (self.nv-1)/2]
            mom_e[iy] += self.v[(self.nv-1)/2]**2 * f[ix, (self.nv-1)/2]
                
            mom_n[iy] *= self.hv
            mom_u[iy] *= self.hv / mom_n[iy]
            mom_e[iy] *= self.hv / mom_n[iy]
            
            a1[iy] = mom_u[iy]
            a2[iy] = mom_e[iy] - mom_u[iy]**2



    def potential_to_hamiltonian(self, Vec P, Vec H):
        (xs, xe), = self.dax.getRanges()
        
        p = self.dax.getVecArray(P)
        h = self.da1.getVecArray(H)
        
        for j in np.arange(0, self.nv):
            h[xs:xe, j] = p[xs:xe]

