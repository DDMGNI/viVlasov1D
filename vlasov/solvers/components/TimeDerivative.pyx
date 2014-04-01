'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc


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
    
    
    @staticmethod
    def create(str  type not None,
               VIDA da1  not None,
               Grid grid not None):
        
        if type == 'point':
            return TimeDerivative(da1, grid) 
        elif type == 'midpoint':
            return TimeDerivativeMidpoint(da1, grid)
        elif type == 'simpson':
            return TimeDerivativeSimpson(da1, grid)
        elif type == 'arakawa_J1':
            return TimeDerivativeArakawaJ1(da1, grid)
        elif type == 'arakawa_J2':
            return TimeDerivativeArakawaJ2(da1, grid)
        elif type == 'arakawa_J4':
            return TimeDerivativeArakawaJ4(da1, grid)
        else:
            return None
        

    cdef void function(self, Vec F, Vec Y):
        Y.axpy(self.grid.ht_inv, F)
        
    
    cdef void jacobian(self, Mat J):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double time_fac = 1.0  / self.grid.ht
        
        row = Mat.Stencil()
        row.field = 0
        
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                row.index = (i,j)
                
                J.setValueStencil(row, row, time_fac, addv=PETSc.InsertMode.ADD_VALUES)



cdef class TimeDerivativeMidpoint(TimeDerivative):
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void function(self, Vec F, Vec Y):
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double time_fac = self.grid.ht_inv / 16.
        
        cdef double[:,:] f = self.da1.getLocalArray(F, self.localF)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            for i in range(xs, xe):
                ix = i-xs+self.grid.stencil
                iy = i-xs
                
                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    y[iy, jy] += self.grid.ht_inv * f[ix, jx]
                else:
                    y[iy, jy] += ( 1. * f[ix-1, jx-1] + 2. * f[ix, jx-1] + 1. * f[ix+1, jx-1] \
                                 + 2. * f[ix-1, jx  ] + 4. * f[ix, jx  ] + 2. * f[ix+1, jx  ] \
                                 + 1. * f[ix-1, jx+1] + 2. * f[ix, jx+1] + 1. * f[ix+1, jx+1] \
                                 ) * time_fac
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void jacobian(self, Mat J):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double time_fac = self.grid.ht_inv / 16.
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        row.field = 0
        col.field = 0
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            for i in range(xs, xe):
                ix = i-xs+self.grid.stencil
        
                row.index = (i,j)
                
                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    J.setValueStencil(row, row, self.grid.ht_inv, addv=PETSc.InsertMode.ADD_VALUES)
                else:
                    for index, value in [
                            ((i-1, j-1), 1.),
                            ((i-1, j  ), 2.),
                            ((i-1, j+1), 1.),
                            ((i,   j-1), 2.),
                            ((i,   j  ), 4.),
                            ((i,   j+1), 2.),
                            ((i+1, j-1), 1.),
                            ((i+1, j  ), 2.),
                            ((i+1, j+1), 1.),
                        ]:
    
                        col.index = index
                        J.setValueStencil(row, col, value * time_fac, addv=PETSc.InsertMode.ADD_VALUES)
                        


cdef class TimeDerivativeSimpson(TimeDerivative):
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void function(self, Vec F, Vec Y):
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double time_fac = self.grid.ht_inv / 36.
        
        cdef double[:,:] f = self.da1.getLocalArray(F, self.localF)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            for i in range(xs, xe):
                ix = i-xs+self.grid.stencil
                iy = i-xs
                    
                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    y[iy, jy] += self.grid.ht_inv * f[ix, jx]
                else:
                    y[iy, jy] += ( 1. * f[ix-1, jx-1] + 4. * f[ix, jx-1] + 1. * f[ix+1, jx-1] \
                                 + 4. * f[ix-1, jx  ] +16. * f[ix, jx  ] + 4. * f[ix+1, jx  ] \
                                 + 1. * f[ix-1, jx+1] + 4. * f[ix, jx+1] + 1. * f[ix+1, jx+1] \
                                 ) * time_fac
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void jacobian(self, Mat J):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double time_fac = self.grid.ht_inv / 16.
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        row.field = 0
        col.field = 0
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            for i in range(xs, xe):
                ix = i-xs+self.grid.stencil
        
                row.index = (i,j)
                
                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    J.setValueStencil(row, row, self.grid.ht_inv, addv=PETSc.InsertMode.ADD_VALUES)
                else:
                    for index, value in [
                            ((i-1, j-1),  1.),
                            ((i-1, j  ),  4.),
                            ((i-1, j+1),  1.),
                            ((i,   j-1),  4.),
                            ((i,   j  ), 16.),
                            ((i,   j+1),  4.),
                            ((i+1, j-1),  1.),
                            ((i+1, j  ),  4.),
                            ((i+1, j+1),  1.),
                        ]:
    
                        col.index = index
                        J.setValueStencil(row, col, value * time_fac, addv=PETSc.InsertMode.ADD_VALUES)
                      
      
    
cdef class TimeDerivativeArakawaJ1(TimeDerivativeMidpoint):
    pass
    

cdef class TimeDerivativeArakawaJ2(TimeDerivative):
    pass
    
    
    
cdef class TimeDerivativeArakawaJ4(TimeDerivative):
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void function(self, Vec F, Vec Y):
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double time_fac = self.grid.ht_inv / 64.
        
        cdef double[:,:] f = self.da1.getLocalArray(F, self.localF)
        cdef double[:,:] y = self.da1.getGlobalArray(Y)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            for i in range(xs, xe):
                ix = i-xs+self.grid.stencil
                iy = i-xs
                    
                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    y[iy, jy] += self.grid.ht_inv * f[ix, jx]
                else:
                    y[iy, jy] += (                                           1. * f[ix, jx-2] \
                                                      + 2. * f[ix-1, jx-1] + 8. * f[ix, jx-1] + 2. * f[ix+1, jx-1] \
                                 + 1. * f[ix-2, jx  ] + 8. * f[ix-1, jx  ] +20. * f[ix, jx  ] + 8. * f[ix+1, jx  ] + 1. * f[ix+2, jx  ] \
                                                      + 2. * f[ix-1, jx+1] + 8. * f[ix, jx+1] + 2. * f[ix+1, jx+1] \
                                                                           + 1. * f[ix, jx+2] \
                                 ) * time_fac


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void jacobian(self, Mat J):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double time_fac = self.grid.ht_inv / 16.
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        row.field = 0
        col.field = 0
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            for i in range(xs, xe):
                ix = i-xs+self.grid.stencil
        
                row.index = (i,j)
                
                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    J.setValueStencil(row, row, self.grid.ht_inv, addv=PETSc.InsertMode.ADD_VALUES)
                else:
                    for index, value in [
                            ((i-2, j  ),  1.),
                            ((i-1, j-1),  2.),
                            ((i-1, j  ),  8.),
                            ((i-1, j+1),  2.),
                            ((i,   j-2),  1.),
                            ((i,   j-1),  8.),
                            ((i,   j  ), 20.),
                            ((i,   j+1),  8.),
                            ((i,   j+2),  1.),
                            ((i+1, j-1),  2.),
                            ((i+1, j  ),  8.),
                            ((i+1, j+1),  2.),
                            ((i+2, j  ),  1.),
                        ]:
    
                        col.index = index
                        J.setValueStencil(row, col, value * time_fac, addv=PETSc.InsertMode.ADD_VALUES)
                      
      
