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
                 config    not None,
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
    
        # set collision operator functions
        if config.is_dissipation_collisions():
            self.collision_operator_function = &self.collT_function
            self.collision_operator_jacobian = &self.collT_jacobian
        else:
            self.collision_operator_function = NULL
            self.collision_operator_jacobian = NULL
            
        
    cdef void call_function(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor):
        if self.collision_operator_function != NULL:
            self.collision_operator_function(self, F, Y, N, U, E, A, factor)
    
    
    cdef void call_jacobian(self, Mat J, Vec N, Vec U, Vec E, Vec A, double factor):
        if self.collision_operator_jacobian != NULL:
            self.collision_operator_jacobian(self, J, N, U, E, A, factor)
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void collT_function(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor):
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
            
                        coll_drag = ( (v[j+1] - u[i] / n[i]) * f[ix, jx+1] - (v[j-1] - u[i] / n[i]) * f[ix, jx-1] ) * a[i]
                        coll_diff = ( f[ix, jx+1] - 2. * f[ix, jx] + f[ix, jx-1] )
                        
                        y[iy, jy] -= factor * self.coll_freq * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5
                        y[iy, jy] -= factor * self.coll_freq * self.coll_diff * coll_diff * self.grid.hv2_inv
                        
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void collE_function(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor):
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
            
                        coll_drag = ( (v[j+1] - u[i] / n[i]) * f[ix, jx+1] - (v[j-1] - u[i] / n[i]) * f[ix, jx-1] ) 
                        coll_diff = ( f[ix, jx+1] - 2. * f[ix, jx] + f[ix, jx-1] ) / a[i]
                        
                        y[iy, jy] -= factor * self.coll_freq * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5
                        y[iy, jy] -= factor * self.coll_freq * self.coll_diff * coll_diff * self.grid.hv2_inv
                        


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void collT_jacobian(self, Mat J, Vec N, Vec U, Vec E, Vec A, double factor):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        cdef double[:] v, u, a
        
        cdef double coll_drag_fac = - factor * self.coll_freq * self.coll_drag * self.grid.hv_inv * 0.5
        cdef double coll_diff_fac = - factor * self.coll_freq * self.coll_diff * self.grid.hv2_inv
        
        if self.coll_freq > 0.:
            v = self.grid.v
            n = N.getArray()
            u = U.getArray()
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
                                ((i,   j-1), - coll_drag_fac * ( v[j-1] - u[ix] / n[i] ) * a[ix] \
                                             + coll_diff_fac),
                                ((i,   j  ), - 2. * coll_diff_fac),
                                ((i,   j+1), + coll_drag_fac * ( v[j+1] - u[ix] / n[i] ) * a[ix] \
                                             + coll_diff_fac),
                            ]:
                            
                            col.index = index
                            J.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void collE_jacobian(self, Mat J, Vec N, Vec U, Vec E, Vec A, double factor):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        cdef double[:] v, u, a
        
        cdef double coll_drag_fac = - factor * self.coll_freq * self.coll_drag * self.grid.hv_inv * 0.5
        cdef double coll_diff_fac = - factor * self.coll_freq * self.coll_diff * self.grid.hv2_inv
        
        if self.coll_freq > 0.:
            v = self.grid.v
            n = N.getArray()
            u = U.getArray()
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
                                ((i,   j-1), - coll_drag_fac * ( v[j-1] - u[ix] / n[i] ) \
                                             + coll_diff_fac / a[ix]),
                                ((i,   j  ), - 2. * coll_diff_fac / a[ix]),
                                ((i,   j+1), + coll_drag_fac * ( v[j+1] - u[ix] / n[i] ) * a[ix] \
                                             + coll_diff_fac / a[ix]),
                            ]:
                            
                            col.index = index
                            J.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
