'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np


cdef class Regularisation(object):
    '''
    
    '''
    
    def __init__(self,
                 VIDA da1  not None,
                 Grid grid not None,
                 double epsilon=0.):
        '''
        Constructor
        '''
        
        # distributed arrays and grid
        self.da1  = da1
        self.grid = grid
        
        # regularisation parameters
        self.epsilon = epsilon
        
        # create local vectors
        self.localF = da1.createLocalVec()
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef regularisation(self, Vec F, Vec Y, double factor):
        '''
        Collision Operator
        '''
        
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double[:,:] f, y
        
        
        if self.epsilon > 0.:
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
            
                        y[iy, jy] += factor * self.epsilon * self.grid.ht * self.grid.hx2_inv * ( 2. * f[ix, jx] - f[ix+1, jx] - f[ix-1, jx] )
                        y[iy, jy] += factor * self.epsilon * self.grid.ht * self.grid.hv2_inv * ( 2. * f[ix, jx] - f[ix, jx+1] - f[ix, jx-1] )
                        
