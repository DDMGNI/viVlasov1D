'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc


cdef class CollisionOperator(object):
    '''
    
    '''
    
    def __init__(self,
                 object da1  not None,
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
        

    @staticmethod
    def create(str    type not None,
               object da1  not None,
               Grid   grid not None,
               double coll_freq=0.,
               double coll_diff=1.,
               double coll_drag=1.):
        
        
        if type == 'collt':
            return CollisionOperatorT(da1, grid, coll_freq, coll_diff, coll_drag)
        elif type == 'colle':
            return CollisionOperatorE(da1, grid, coll_freq, coll_diff, coll_drag)
        else:
            return CollisionOperator(da1, grid, coll_freq, coll_diff, coll_drag)
        
    
    cdef void function(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor):
        pass
    
    cdef void jacobian(self, Mat J, Vec N, Vec U, Vec E, Vec A, double factor):
        pass
    
    
    
cdef class CollisionOperatorT(CollisionOperator):

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void function(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor):
        '''
        Collision Operator
        '''
        
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double coll_drag_fac = - factor * self.coll_freq * self.coll_drag * self.grid.hv_inv * 0.5
        cdef double coll_diff_fac = - factor * self.coll_freq * self.coll_diff * self.grid.hv2_inv
        
        cdef double[:]   v, n, u, a
        cdef double[:,:] f, y
        
        
        if self.coll_freq != 0. and factor != 0.:
            v = self.grid.v
            n = N.getArray()
            u = U.getArray() / n
            a = A.getArray()
            
            f = getLocalArray(self.da1, F, self.localF)
            y = getGlobalArray(self.da1, Y)
        
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
                        
                        y[iy, jy] += coll_drag_fac * coll_drag + coll_diff_fac * coll_diff
                        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void jacobian(self, Mat J, Vec N, Vec U, Vec E, Vec A, double factor):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        cdef double[:] v, n, u, a
        
        cdef double coll_drag_fac = - factor * self.coll_freq * self.coll_drag * self.grid.hv_inv * 0.5
        cdef double coll_diff_fac = - factor * self.coll_freq * self.coll_diff * self.grid.hv2_inv
        
        if self.coll_freq != 0. and factor != 0.:
            v = self.grid.v
            n = N.getArray()
            u = U.getArray() / n
            a = A.getArray()
        
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
                                ((i,   j-1), - coll_drag_fac * (v[j-1] - u[i]) * a[i] \
                                             + coll_diff_fac),
                                ((i,   j  ), - 2. * coll_diff_fac),
                                ((i,   j+1), + coll_drag_fac * (v[j+1] - u[i]) * a[i] \
                                             + coll_diff_fac),
                            ]:
                            
                            col.index = index
                            J.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                        


cdef class CollisionOperatorE(CollisionOperator):
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void function(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor):
        '''
        Collision Operator
        '''
        
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double coll_drag_fac = - factor * self.coll_freq * self.coll_drag * self.grid.hv_inv * 0.5
        cdef double coll_diff_fac = - factor * self.coll_freq * self.coll_diff * self.grid.hv2_inv
        
        cdef double[:]   v, n, u, a
        cdef double[:,:] f, y
        
        
        if self.coll_freq != 0. and factor != 0.:
            v = self.grid.v
            n = N.getArray()
            u = U.getArray() / n
            a = A.getArray()
            
            f = getLocalArray(self.da1, F, self.localF)
            y = getGlobalArray(self.da1, Y)
        
            (xs, xe), (ys, ye) = self.da1.getRanges()
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys
                
                if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                    for i in range(xs, xe):
                        ix = i-xs+self.grid.stencil
                        iy = i-xs
            
                        coll_drag = (v[j+1] - u[i]) * f[ix, jx+1] - (v[j-1] - u[i]) * f[ix, jx-1]
                        coll_diff = ( f[ix, jx+1] - 2. * f[ix, jx] + f[ix, jx-1] ) / a[i]
                        
                        y[iy, jy] += coll_drag_fac * coll_drag + coll_diff_fac * coll_diff


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void jacobian(self, Mat J, Vec N, Vec U, Vec E, Vec A, double factor):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        cdef double[:] v, n, u, a
        
        cdef double coll_drag_fac = - factor * self.coll_freq * self.coll_drag * self.grid.hv_inv * 0.5
        cdef double coll_diff_fac = - factor * self.coll_freq * self.coll_diff * self.grid.hv2_inv
        
        
        if self.coll_freq != 0. and factor != 0.:
            v = self.grid.v
            n = N.getArray()
            u = U.getArray() / n
            a = A.getArray()
        
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
                                ((i,   j-1), - coll_drag_fac * (v[j-1] - u[i]) \
                                             + coll_diff_fac / a[i]),
                                ((i,   j  ), - 2. * coll_diff_fac / a[i]),
                                ((i,   j+1), + coll_drag_fac * (v[j+1] - u[i]) \
                                             + coll_diff_fac / a[i]),
                            ]:
                            
                            col.index = index
                            J.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)

