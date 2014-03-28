'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from libc.math cimport exp, pow, sqrt


cdef class TimeDerivative(object):
    '''
    
    '''
    
    def __init__(self,
                 VIDA da1  not None,
                 Grid grid not None):
        '''
        Constructor
        '''
        
        # distributed array and grid
        self.da1  = da1
        self.grid = grid
        
        # create local vectors
        self.localF = da1.createLocalVec()
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void arakawa_J1(self, Vec F, Vec Y):
        self.midpoint(F,Y)
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void arakawa_J2(self, Vec F, Vec Y):
        pass
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void arakawa_J4(self, Vec F, Vec Y):
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double time_fac = self.grid.ht_inv / 64.
        
        cdef double[:,:] f = self.da1.getLocalArray(F, self.localF)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                y[iy, jy] += self.grid.ht_inv * f[ix, jx]
            else:
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
                    
                    y[iy, jy] += (                                           1. * f[ix, jx-2] \
                                                      + 2. * f[ix-1, jx-1] + 8. * f[ix, jx-1] + 2. * f[ix+1, jx-1] \
                                 + 1. * f[ix-2, jx  ] + 8. * f[ix-1, jx  ] +20. * f[ix, jx  ] + 8. * f[ix+1, jx  ] + 1. * f[ix+2, jx  ] \
                                                      + 2. * f[ix-1, jx+1] + 8. * f[ix, jx+1] + 2. * f[ix+1, jx+1] \
                                                                           + 1. * f[ix, jx+2] \
                                 ) * time_fac

    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void midpoint(self, Vec F, Vec Y):
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double time_fac = self.grid.ht_inv / 16.
        
        cdef double[:,:] f = self.da1.getLocalArray(F, self.localF)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                y[iy, jy] += self.grid.ht_inv * f[ix, jx]
            else:
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
                    
                    y[iy, jy] += ( 1. * f[ix-1, jx-1] + 2. * f[ix, jx-1] + 1. * f[ix+1, jx-1] \
                                 + 2. * f[ix-1, jx  ] + 4. * f[ix, jx  ] + 2. * f[ix+1, jx  ] \
                                 + 1. * f[ix-1, jx+1] + 2. * f[ix, jx+1] + 1. * f[ix+1, jx+1] \
                                 ) * time_fac
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void simpson(self, Vec F, Vec Y):
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double time_fac = self.grid.ht_inv / 36.
        
        cdef double[:,:] f = self.da1.getLocalArray(F, self.localF)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                y[iy, jy] += self.grid.ht_inv * f[ix, jx]
            else:
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
                    
                    y[iy, jy] += ( 1. * f[ix-1, jx-1] + 4. * f[ix, jx-1] + 1. * f[ix+1, jx-1] \
                                 + 4. * f[ix-1, jx  ] +16. * f[ix, jx  ] + 4. * f[ix+1, jx  ] \
                                 + 1. * f[ix-1, jx+1] + 4. * f[ix, jx+1] + 1. * f[ix+1, jx+1] \
                                 ) * time_fac
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void time_derivative(self, Vec F, Vec Y):
        Y.axpy(self.grid.ht_inv, F)
