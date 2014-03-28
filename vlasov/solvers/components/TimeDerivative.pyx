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
    cpdef arakawa_J1(self, Vec F, Vec Y):
        pass
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef arakawa_J2(self, Vec F, Vec Y):
        pass
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef arakawa_J4(self, Vec F, Vec Y):
        pass
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef midpoint(self, Vec F, Vec Y):
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double[:,:] f = self.da1.getLocalArray(F, self.localF)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
                    
                    y[iy, jy] += f[ix, jx] * self.grid.ht_inv 
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Simpson(self, Vec F, Vec Y):
        pass
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef time_derivative(self, Vec F, Vec Y):
        Y.axpy(self.grid.ht_inv, F)
        
#         cdef int i, j, ix, iy, jx, jy
#         cdef int xs, xe, ys, ye
#         
#         cdef double[:,:] f = self.da1.getGlobalArray(F)
#         cdef double[:,:] y = self.da1.getGlobalArray(Y)
#         
#         (xs, xe), (ys, ye) = self.da1.getRanges()
#         
#         for j in range(ys, ye):
#             jy = j-ys
#             
#             if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
#                 for i in range(xs, xe):
#                     iy = i-xs
#                     
#                     y[iy, jy] += f[iy, jy] * self.grid.ht_inv 
            
