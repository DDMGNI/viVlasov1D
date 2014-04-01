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
                 config    not None,
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
        
        # set time derivative functions
        if config.is_averaging_operator_none():
            self.time_derivative_function = &self.point
            self.time_derivative_jacobian = &self.point_jacobian
        elif config.is_averaging_operator_midpoint():
            self.time_derivative_function = &self.midpoint
#             self.time_derivative_jacobian = &self.midpoint_jacobian
            self.time_derivative_jacobian = NULL
        elif config.is_averaging_operator_simpson():
            self.time_derivative_function = &self.simpson
#             self.time_derivative_jacobian = &self.simpson_jacobian
            self.time_derivative_jacobian = NULL
        elif config.is_averaging_operator_arakawa_J1():
            self.time_derivative_function = &self.arakawa_J1
#             self.time_derivative_jacobian = &self.arakawa_J1_jacobian
            self.time_derivative_jacobian = NULL
        elif config.is_averaging_operator_arakawa_J2():
            self.time_derivative_function = &self.arakawa_J2
#             self.time_derivative_jacobian = &self.arakawa_J2_jacobian
            self.time_derivative_jacobian = NULL
        elif config.is_averaging_operator_arakawa_J4():
            self.time_derivative_function = &self.arakawa_J4
#             self.time_derivative_jacobian = &self.arakawa_J4_jacobian
            self.time_derivative_jacobian = NULL
        else:
            self.time_derivative_function = NULL
            self.time_derivative_jacobian = NULL
        

    cdef void call_function(self, Vec F, Vec Y):
        print("time derivative in")
        self.time_derivative_function(self, F, Y)
        print("time derivative out")
        
    
    cdef void call_jacobian(self, Mat J):
        self.time_derivative_jacobian(self, J)
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void point(self, Vec F, Vec Y):
        Y.axpy(self.grid.ht_inv, F)
    
    
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
    cdef void point_jacobian(self, Mat J):
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


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void midpoint_jacobian(self, Mat J):
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
                        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void simpson_jacobian(self, Mat J):
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
                      
      
    cdef void arakawa_J1_jacobian(self, Mat J):
        self.midpoint_jacobian(J)
    
    
    cdef void arakawa_J2_jacobian(self, Mat J):
        pass
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void arakawa_J4_jacobian(self, Mat J):
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
                      
      
