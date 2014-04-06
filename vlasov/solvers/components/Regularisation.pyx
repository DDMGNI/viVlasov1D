'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc


cdef class Regularisation(object):
    '''
    
    '''
    
    def __init__(self,
                 config    not None,
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
        
            
    
    cdef void function(self, Vec F, Vec Y, double factor):
        if self.epsilon != 0.:
            self.call_regularisation_function(self, F, Y, factor)

    
    cdef void jacobian(self, Mat J, double factor):
        if self.epsilon != 0.:
            self.call_regularisation_jacobiann(self, J, factor)

    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void regularisation_function(self, Vec F, Vec Y, double factor):
        '''
        Collision Operator
        '''
        
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double regularisation_factor = factor * self.epsilon * self.grid.ht * self.grid.hx2_inv
        
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
            
                        y[iy, jy] += regularisation_factor * ( 2. * f[ix, jx] - f[ix+1, jx] - f[ix-1, jx] )
                        y[iy, jy] += regularisation_factor * ( 2. * f[ix, jx] - f[ix, jx+1] - f[ix, jx-1] )
                        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void regularisation_jacobian(self, Mat J, double factor):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        cdef double regularisation_factor = factor * self.epsilon * self.grid.ht * self.grid.hx2_inv
        
        if self.epsilon > 0.:
            (xs, xe), (ys, ye) = self.da1.getRanges()
            
            row = Mat.Stencil()
            col = Mat.Stencil()
            row.field = 0
            col.field = 0
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys
    
                if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                    for i in range(xs, xe):
                        ix = i-xs+self.grid.stencil
                
                        row.index = (i,j)
                    
                        for index, value in [
                            ((i-1, j  ), - 1. * regularisation_factor),
                            ((i,   j-1), - 1. * regularisation_factor),
                            ((i,   j  ), + 4. * regularisation_factor),
                            ((i,   j+1), - 1. * regularisation_factor),
                            ((i+1, j  ), - 1. * regularisation_factor),
                            ]:
    
                            col.index = index
                            J.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                        

