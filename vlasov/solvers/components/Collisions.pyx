'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from libc.math cimport exp, pow, sqrt


cdef class Collisions(object):
    '''
    
    '''
    
    def __init__(self,
                 VIDA da1  not None,
                 Grid grid not None,
                 double coll_freq=0.,
                 double coll_diff=1.,
                 double coll_drag=1.):
        '''
        Constructor
        '''
        
        # distributed arrays and grid
        self.da1  = da1
        self.grid = grid
        
        # collision parameters
        self.coll_freq = coll_freq
        self.coll_diff = coll_diff
        self.coll_drag = coll_drag
        
        # create local vectors
        self.localF = da1.createLocalVec()
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void collT(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor):
        '''
        Collision Operator
        '''
        
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double[:]   v, up, uh, ap, ah
        cdef double[:,:] f, y
        
        
        if self.coll_freq > 0.:
            v = self.grid.v
            u = U.getArray()
            a = A.getArray()
            f = self.da1.getLocalArray(F, self.localF)
            y = self.da1.getGlobalArray(Y)
        
            (xs, xe), (ys, ye) = self.da1.getRanges()
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys
                
                if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                    for i in range(xs, xe):
                        ix = i-xs+self.grid.stencil
                        iy = i-xs
            
                        coll_drag = ( (v[j+1] - u[i]) * f[ix, jx+1] - (v[j-1] - u[i]) * f[ix, jx-1] ) * a[i]
                        coll_diff = ( f[ix, jx+1] - 2. * f[ix, jx] + f[ix, jx-1] )
                        
                        y[iy, jy] -= factor * self.coll_freq * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5
                        y[iy, jy] -= factor * self.coll_freq * self.coll_diff * coll_diff * self.grid.hv2_inv
                        
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void collE(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor):
        '''
        Collision Operator
        '''
        
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double[:]   v, up, uh, ap, ah
        cdef double[:,:] f, y
        
        
        if self.coll_freq > 0.:
            v = self.grid.v
            n = N.getArray()
            u = U.getArray()
            a = A.getArray()
            f = self.da1.getLocalArray(F, self.localF)
            y = self.da1.getGlobalArray(Y)
        
            (xs, xe), (ys, ye) = self.da1.getRanges()
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys
                
                if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                    for i in range(xs, xe):
                        ix = i-xs+self.grid.stencil
                        iy = i-xs
            
                        coll_drag = ( (n[i] * v[j+1] - u[i]) * f[ix, jx+1] - (n[i] * v[j-1] - u[i]) * f[ix, jx-1] ) * a[i]
                        coll_diff = ( f[ix, jx+1] - 2. * f[ix, jx] + f[ix, jx-1] )
                        
                        y[iy, jy] -= factor * self.coll_freq * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5
                        y[iy, jy] -= factor * self.coll_freq * self.coll_diff * coll_diff * self.grid.hv2_inv
                        
