'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc


cdef class PETScSmoother(object):
    '''
    
    '''
    
    def __init__(self, VIDA da_coar, VIDA da_fine,
                 np.uint64_t nx_coar, np.uint64_t nv_coar,
                 np.uint64_t nx_fine, np.uint64_t nv_fine):
        '''
        Constructor
        '''
        
        assert nx_coar == nx_fine / 2
        assert (nv_coar-1 == (nv_fine-1) / 2) or (nv_coar == nv_fine == 1)
        
        # distributed array
        self.da_coar = da_coar
        self.da_fine = da_fine
        
        # grid
        self.grid.nx_coar = nx_coar
        self.grid.nv_coar = nv_coar
        
        self.grid.nx_fine = nx_fine
        self.grid.nv_fine = nv_fine
        
        # create local vectors
        self.localX_coar = da_coar.createLocalVec()
        self.localX_fine = da_fine.createLocalVec()
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def smoothen(self, Vec Xcoar, Vec Xfine):
        cdef np.int64_t i, j, ix, iy
        cdef np.int64_t xe_coar, xs_coar
        cdef np.int64_t xe_fine, xs_fine
        
        (xs_coar, xe_coar), = self.da_coar.getRanges()
        (xs_fine, xe_fine), = self.da_fine.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=1] x_coar_1d
        cdef np.ndarray[np.float64_t, ndim=1] x_fine_1d
        cdef np.ndarray[np.float64_t, ndim=2] x_coar_2d
        cdef np.ndarray[np.float64_t, ndim=2] x_fine_2d
        
        if self.grid.nv_coar == self.grid.nv_fine == 1:
            x_coar_1d = self.da_coar.getLocalArray(Xcoar, self.localX_coar)
            x_fine_1d = self.da_fine.getGlobalArray(Xfine)
        else:
            x_coar_2d = self.da_coar.getLocalArray(Xcoar, self.localX_coar)
            x_fine_2d = self.da_fine.getGlobalArray(Xfine)
        
        
        for i in range(xs_coar, xe_coar):
            ix = (i-xs_coar) + 2
            iy = (i-xs_coar) * 2
            
            if self.grid.nv_coar == self.grid.nv_fine == 1:
                x_fine_1d[iy  ] = x_coar_1d[ix]
                x_fine_1d[iy+1] = 0.5 * ( x_coar_1d[ix] + x_coar_1d[ix+1] )
            
            else:
                x_fine_2d[iy,   -2] = x_coar_2d[ix, -2]
                x_fine_2d[iy+1, -2] = 0.5  * ( x_coar_2d[ix, -2] + x_coar_2d[ix+1, -2] )
                
                x_fine_2d[iy,   -1] = 0.
                x_fine_2d[iy+1, -1] = 0.
                
                for j in range(0, self.grid.nv_coar-1):
                    jx = j
                    jy = j * 2
                    
                    x_fine_2d[iy,  jy  ] = x_coar_2d[ix, jx]
                    x_fine_2d[iy,  jy+1] = 0.5  * ( x_coar_2d[ix,   jx] + x_coar_2d[ix,   jx+1] )
                    x_fine_2d[iy+1,jy  ] = 0.5  * ( x_coar_2d[ix,   jx] + x_coar_2d[ix+1, jx  ] )
                    x_fine_2d[iy+1,jy+1] = 0.25 * ( x_coar_2d[ix,   jx] + x_coar_2d[ix,   jx+1] \
                                                  + x_coar_2d[ix+1, jx] + x_coar_2d[ix+1, jx+1] )
                
        
                
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def coarsen(self, Vec Xcoar, Vec Xfine):
        cdef np.int64_t i, j, ix, iy
        cdef np.int64_t xe_coar, xs_coar
        cdef np.int64_t xe_fine, xs_fine
        
        (xs_coar, xe_coar), = self.da_coar.getRanges()
        (xs_fine, xe_fine), = self.da_fine.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=1] x_coar_1d
        cdef np.ndarray[np.float64_t, ndim=1] x_fine_1d
        cdef np.ndarray[np.float64_t, ndim=2] x_coar_2d
        cdef np.ndarray[np.float64_t, ndim=2] x_fine_2d
        
        if self.grid.nv_coar == self.grid.nv_fine == 1:
            x_coar_1d = self.da_coar.getGlobalArray(Xcoar)
            x_fine_1d = self.da_fine.getLocalArray(Xfine, self.localX_fine)
        else:
            x_coar_2d = self.da_coar.getGlobalArray(Xcoar)
            x_fine_2d = self.da_fine.getLocalArray(Xfine, self.localX_fine)

        
        for i in range(xs_coar, xe_coar):
            ix = (i-xs_coar)*2 + 2
            iy = (i-xs_coar)
            
            if self.grid.nv_coar == self.grid.nv_fine == 1:
                x_coar_1d[iy] = 0.25 * ( x_fine_1d[ix-1] + 2. * x_fine_1d[ix] + x_fine_1d[ix+1] )
            
            else:
                x_coar_2d[iy, 0] = 0.
                x_coar_2d[iy,-1] = 0.
                
                for j in range(1, self.grid.nv_coar-1):
                    jx = j*2
                    jy = j
                    
                    x_coar_2d[iy,jy] = ( 4. * x_fine_2d[ix, jx] + 2. * x_fine_2d[ix+1, jx  ] + 2. * x_fine_2d[ix,   jx+1] \
                                                                + 2. * x_fine_2d[ix-1, jx  ] + 2. * x_fine_2d[ix,   jx-1] \
                                                                +      x_fine_2d[ix+1, jx+1] +      x_fine_2d[ix+1, jx-1] \
                                                                +      x_fine_2d[ix-1, jx+1] +      x_fine_2d[ix-1, jx-1] ) / 16.
                    
                
            
