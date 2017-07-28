'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc


cdef class PoissonBracket(object):
    '''
    
    '''
    
    def __init__(self,
                 object da1  not None,
                 Grid   grid not None):
        '''
        Constructor
        '''
        
        # distributed array and grid
        self.da1  = da1
        self.grid = grid
        
        # create local vectors
        self.localF = da1.createLocalVec()
        self.localH = da1.createLocalVec()
        
        
    @staticmethod
    def create(str    type not None,
               object da1  not None,
               Grid   grid not None):
        
        if type == 'simpson':
            return PoissonBracketSimpson(da1, grid)
        elif type == 'arakawaj1':
            return PoissonBracketArakawaJ1(da1, grid)
        elif type == 'arakawaj2':
            return PoissonBracketArakawaJ2(da1, grid)
        elif type == 'arakawaj4':
            return PoissonBracketArakawaJ4(da1, grid)
        else:
            return PoissonBracket(da1, grid)
        

    
    cdef void function(self, Vec F, Vec H, Vec Y, double factor):
        print("ERROR: function not implemented!")
    
    cdef void jacobian(self, Mat J, Vec H, double factor):
        print("ERROR: function not implemented!")

    cdef void poisson_bracket_array(self, double[:,:] x, double[:,:] h, double[:,:] y, double factor):
        print("ERROR: function not implemented!")
 
    cdef double poisson_bracket_point(self, double[:,:] f, double[:,:] h, int i, int j):
        print("ERROR: function not implemented!")
    


cdef class PoissonBracketArakawaJ1(PoissonBracket):
    
    cdef void function(self, Vec F, Vec H, Vec Y, double factor):
        cdef double[:,:] f = getLocalArray(self.da1, F, self.localF)
        cdef double[:,:] h = getLocalArray(self.da1, H, self.localH)
        cdef double[:,:] y = getGlobalArray(self.da1, Y)
        
        self.poisson_bracket_array(f, h, y, factor)
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void poisson_bracket_array(self, double[:,:] x, double[:,:] h, double[:,:] y, double factor):
        
        cdef double jpp, jpc, jcp, result, arakawa_factor
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        arakawa_factor = self.grid.hx_inv * self.grid.hv_inv / 12.
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                # Vlasov equation with Dirichlet Boundary Conditions
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
                    
                    jpp = (x[ix+1, jx  ] - x[ix-1, jx  ]) * (h[ix,   jx+1] - h[ix,   jx-1]) \
                        - (x[ix,   jx+1] - x[ix,   jx-1]) * (h[ix+1, jx  ] - h[ix-1, jx  ])
                    
                    jpc = x[ix+1, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                        - x[ix-1, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                        - x[ix,   jx+1] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                        + x[ix,   jx-1] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp = x[ix+1, jx+1] * (h[ix,   jx+1] - h[ix+1, jx  ]) \
                        - x[ix-1, jx-1] * (h[ix-1, jx  ] - h[ix,   jx-1]) \
                        - x[ix-1, jx+1] * (h[ix,   jx+1] - h[ix-1, jx  ]) \
                        + x[ix+1, jx-1] * (h[ix+1, jx  ] - h[ix,   jx-1])
        
                    y[iy, jy] += factor * (jpp + jpc + jcp) * arakawa_factor
                    

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double poisson_bracket_point(self, double[:,:] f, double[:,:] h, int i, int j):
        '''
        Arakawa Bracket J1 (second order)
        '''
        
        cdef double jpp, jpc, jcp
        
        jpp = (f[i+1, j  ] - f[i-1, j  ]) * (h[i,   j+1] - h[i,   j-1]) \
            - (f[i,   j+1] - f[i,   j-1]) * (h[i+1, j  ] - h[i-1, j  ])
        
        jpc = f[i+1, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
            - f[i-1, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
            - f[i,   j+1] * (h[i+1, j+1] - h[i-1, j+1]) \
            + f[i,   j-1] * (h[i+1, j-1] - h[i-1, j-1])
        
        jcp = f[i+1, j+1] * (h[i,   j+1] - h[i+1, j  ]) \
            - f[i-1, j-1] * (h[i-1, j  ] - h[i,   j-1]) \
            - f[i-1, j+1] * (h[i,   j+1] - h[i-1, j  ]) \
            + f[i+1, j-1] * (h[i+1, j  ] - h[i,   j-1])
        
        return (jpp + jpc + jcp) * self.grid.hx_inv * self.grid.hv_inv / 12.
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void jacobian(self, Mat J, Vec H, double factor):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        cdef double[:,:] h = getLocalArray(self.da1, H, self.localH)
        
        cdef double arak_fac_J1 = factor * self.grid.hx_inv * self.grid.hv_inv / 12.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        row.field = 0
        col.field = 0
        
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                row.index = (i,j)
                
                if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                    for index, value in [
                            ((i-1, j-1), - (h[ix-1, jx  ] - h[ix,   jx-1]) * arak_fac_J1),
                            ((i-1, j  ), - (h[ix,   jx+1] - h[ix,   jx-1]) * arak_fac_J1 \
                                         - (h[ix-1, jx+1] - h[ix-1, jx-1]) * arak_fac_J1),
                            ((i-1, j+1), - (h[ix,   jx+1] - h[ix-1, jx  ]) * arak_fac_J1),
                            ((i,   j-1), + (h[ix+1, jx  ] - h[ix-1, jx  ]) * arak_fac_J1 \
                                         + (h[ix+1, jx-1] - h[ix-1, jx-1]) * arak_fac_J1),
                            ((i,   j+1), - (h[ix+1, jx  ] - h[ix-1, jx  ]) * arak_fac_J1 \
                                         - (h[ix+1, jx+1] - h[ix-1, jx+1]) * arak_fac_J1),
                            ((i+1, j-1), + (h[ix+1, jx  ] - h[ix,   jx-1]) * arak_fac_J1),
                            ((i+1, j  ), + (h[ix,   jx+1] - h[ix,   jx-1]) * arak_fac_J1 \
                                         + (h[ix+1, jx+1] - h[ix+1, jx-1]) * arak_fac_J1),
                            ((i+1, j+1), + (h[ix,   jx+1] - h[ix+1, jx  ]) * arak_fac_J1),
                        ]:
    
                        col.index = index
                        J.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                        
    

cdef class PoissonBracketArakawaJ2(PoissonBracket):
    
    cdef void function(self, Vec F, Vec H, Vec Y, double factor):
        cdef double[:,:] f = getLocalArray(self.da1, F, self.localF)
        cdef double[:,:] h = getLocalArray(self.da1, H, self.localH)
        cdef double[:,:] y = getGlobalArray(self.da1, Y)
        
        self.poisson_bracket_array(f, h, y, factor)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void poisson_bracket_array(self, double[:,:] x, double[:,:] h, double[:,:] y, double factor):
        
        cdef double jcc, jpc, jcp, result, arakawa_factor
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        arakawa_factor = self.grid.hx_inv * self.grid.hv_inv / 24.
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                # Vlasov equation with Dirichlet Boundary Conditions
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    jcc = (x[ix+1, jx+1] - x[ix-1, jx-1]) * (h[ix-1, jx+1] - h[ix+1, jx-1]) \
                        - (x[ix-1, jx+1] - x[ix+1, jx-1]) * (h[ix+1, jx+1] - h[ix-1, jx-1])
                    
                    jpc = x[ix+2, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                        - x[ix-2, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                        - x[ix,   jx+2] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                        + x[ix,   jx-2] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp = x[ix+1, jx+1] * (h[ix,   jx+2] - h[ix+2, jx  ]) \
                        - x[ix-1, jx-1] * (h[ix-2, jx  ] - h[ix,   jx-2]) \
                        - x[ix-1, jx+1] * (h[ix,   jx+2] - h[ix-2, jx  ]) \
                        + x[ix+1, jx-1] * (h[ix+2, jx  ] - h[ix,   jx-2])
        
                    y[iy, jy] += factor * (jcc + jpc + jcp) * arakawa_factor
                    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double poisson_bracket_point(self, double[:,:] f, double[:,:] h, int i, int j):
        '''
        Arakawa Bracket
        '''
        
        cdef double jcc, jpc, jcp
        
        jcc = (f[i+1, j+1] - f[i-1, j-1]) * (h[i-1, j+1] - h[i+1, j-1]) \
            - (f[i-1, j+1] - f[i+1, j-1]) * (h[i+1, j+1] - h[i-1, j-1])
        
        jpc = f[i+2, j  ] * (h[i+1, j+1] - h[i+1, j-1]) \
            - f[i-2, j  ] * (h[i-1, j+1] - h[i-1, j-1]) \
            - f[i,   j+2] * (h[i+1, j+1] - h[i-1, j+1]) \
            + f[i,   j-2] * (h[i+1, j-1] - h[i-1, j-1])
        
        jcp = f[i+1, j+1] * (h[i,   j+2] - h[i+2, j  ]) \
            - f[i-1, j-1] * (h[i-2, j  ] - h[i,   j-2]) \
            - f[i-1, j+1] * (h[i,   j+2] - h[i-2, j  ]) \
            + f[i+1, j-1] * (h[i+2, j  ] - h[i,   j-2])
        
        return (jcc + jpc + jcp) * self.grid.hx_inv * self.grid.hv_inv / 24.
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void jacobian(self, Mat J, Vec H, double factor):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        cdef double[:,:] h = getLocalArray(self.da1, H, self.localH)
        
        cdef double arak_fac_J2 = factor * self.grid.hx_inv * self.grid.hv_inv / 24.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        row.field = 0
        col.field = 0
        
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil
                jy = j-ys

                row.index = (i,j)
                
                if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                    for index, value in [
                            ((i-2, j  ), - (h[ix-1, jx+1] - h[ix-1, jx-1]) * arak_fac_J2),
                            ((i-1, j-1), - (h[ix-2, jx  ] - h[ix,   jx-2]) * arak_fac_J2 \
                                         - (h[ix-1, jx+1] - h[ix+1, jx-1]) * arak_fac_J2),
                            ((i-1, j+1), - (h[ix,   jx+2] - h[ix-2, jx  ]) * arak_fac_J2 \
                                         - (h[ix+1, jx+1] - h[ix-1, jx-1]) * arak_fac_J2),
                            ((i,   j-2), + (h[ix+1, jx-1] - h[ix-1, jx-1]) * arak_fac_J2),
                            ((i,   j+2), - (h[ix+1, jx+1] - h[ix-1, jx+1]) * arak_fac_J2),
                            ((i+1, j-1), + (h[ix+2, jx  ] - h[ix,   jx-2]) * arak_fac_J2 \
                                         + (h[ix+1, jx+1] - h[ix-1, jx-1]) * arak_fac_J2),
                            ((i+1, j+1), + (h[ix,   jx+2] - h[ix+2, jx  ]) * arak_fac_J2 \
                                         + (h[ix-1, jx+1] - h[ix+1, jx-1]) * arak_fac_J2),
                            ((i+2, j),   + (h[ix+1, jx+1] - h[ix+1, jx-1]) * arak_fac_J2),
                        ]:
    
                        col.index = index
                        J.setValueStencil(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)
                        
    

cdef class PoissonBracketArakawaJ4(PoissonBracket):
    
    cdef PoissonBracket arakawaJ1
    cdef PoissonBracket arakawaJ2
    
    
    def __init__(self,
                 object da1  not None,
                 Grid   grid not None):

        super().__init__(da1, grid)
        
        self.arakawaJ1 = PoissonBracketArakawaJ1(da1, grid)
        self.arakawaJ2 = PoissonBracketArakawaJ2(da1, grid)
    
    
    cdef void function(self, Vec F, Vec H, Vec Y, double factor):
        cdef double[:,:] f = getLocalArray(self.da1, F, self.localF)
        cdef double[:,:] h = getLocalArray(self.da1, H, self.localH)
        cdef double[:,:] y = getGlobalArray(self.da1, Y)
        
        self.poisson_bracket_array(f, h, y, factor)

    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void poisson_bracket_array(self, double[:,:] x, double[:,:] h, double[:,:] y, double factor):
        
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double jpp_J1, jpc_J1, jcp_J1
        cdef double jcc_J2, jpc_J2, jcp_J2
        cdef double result_J1, result_J2
        
        cdef arakawa_factor_J1 = self.grid.hx_inv * self.grid.hv_inv / 12.
        cdef arakawa_factor_J2 = self.grid.hx_inv * self.grid.hv_inv / 24.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                # Vlasov equation with Dirichlet Boundary Conditions
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    jpp_J1 = (x[ix+1, jx  ] - x[ix-1, jx  ]) * (h[ix,   jx+1] - h[ix,   jx-1]) \
                           - (x[ix,   jx+1] - x[ix,   jx-1]) * (h[ix+1, jx  ] - h[ix-1, jx  ])
                    
                    jpc_J1 = x[ix+1, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - x[ix-1, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - x[ix,   jx+1] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + x[ix,   jx-1] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J1 = x[ix+1, jx+1] * (h[ix,   jx+1] - h[ix+1, jx  ]) \
                           - x[ix-1, jx-1] * (h[ix-1, jx  ] - h[ix,   jx-1]) \
                           - x[ix-1, jx+1] * (h[ix,   jx+1] - h[ix-1, jx  ]) \
                           + x[ix+1, jx-1] * (h[ix+1, jx  ] - h[ix,   jx-1])
                    
                    jcc_J2 = (x[ix+1, jx+1] - x[ix-1, jx-1]) * (h[ix-1, jx+1] - h[ix+1, jx-1]) \
                           - (x[ix-1, jx+1] - x[ix+1, jx-1]) * (h[ix+1, jx+1] - h[ix-1, jx-1])
                    
                    jpc_J2 = x[ix+2, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                           - x[ix-2, jx  ] * (h[ix-1, jx+1] - h[ix-1, jx-1]) \
                           - x[ix,   jx+2] * (h[ix+1, jx+1] - h[ix-1, jx+1]) \
                           + x[ix,   jx-2] * (h[ix+1, jx-1] - h[ix-1, jx-1])
                    
                    jcp_J2 = x[ix+1, jx+1] * (h[ix,   jx+2] - h[ix+2, jx  ]) \
                           - x[ix-1, jx-1] * (h[ix-2, jx  ] - h[ix,   jx-2]) \
                           - x[ix-1, jx+1] * (h[ix,   jx+2] - h[ix-2, jx  ]) \
                           + x[ix+1, jx-1] * (h[ix+2, jx  ] - h[ix,   jx-2])
                    
                    result_J1 = (jpp_J1 + jpc_J1 + jcp_J1) * arakawa_factor_J1
                    result_J2 = (jcc_J2 + jpc_J2 + jcp_J2) * arakawa_factor_J2
                    
                    y[iy, jy] += factor * (2. * result_J1 - result_J2)
    
    
    cdef double poisson_bracket_point(self, double[:,:] f, double[:,:] h, int i, int j):
        '''
        Arakawa Bracket 4th order
        '''
        
        return 2.0 * self.arakawaJ1.poisson_bracket_point(f, h, i, j) - self.arakawaJ2.poisson_bracket_point(f, h, i, j)
    
    
    cdef void jacobian(self, Mat J, Vec H, double factor):
        self.arakawaJ1.jacobian(J, H, +2. * factor)
        self.arakawaJ2.jacobian(J, H, -1. * factor)
    

cdef class PoissonBracketSimpson(PoissonBracket):

    cdef void function(self, Vec F, Vec H, Vec Y, double factor):
        cdef double[:,:] f = getLocalArray(self.da1, F, self.localF)
        cdef double[:,:] h = getLocalArray(self.da1, H, self.localH)
        cdef double[:,:] y = getGlobalArray(self.da1, Y)
        
        self.poisson_bracket_array(f, h, y, factor)

    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void poisson_bracket_array(self, double[:,:] x, double[:,:] h, double[:,:] y, double factor):
        
        cdef int i, j, ix, iy, jx, jy
        cdef int xs, xe, ys, ye
        
        cdef double bracket, bracket11, bracket12, bracket21, bracket22
        
        cdef simpson_factor = factor * self.grid.hx_inv * self.grid.hv_inv / 9.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys
            
            if j >= self.grid.stencil and j < self.grid.nv-self.grid.stencil:
                # Vlasov equation with Dirichlet Boundary Conditions
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    bracket22 = ( \
                                  + (x[ix,   jx+2] - x[ix,   jx-2]) * (h[ix-2, jx  ] - h[ix+2, jx  ]) \
                                  + (x[ix+2, jx  ] - x[ix-2, jx  ]) * (h[ix,   jx+2] - h[ix,   jx-2]) \
                                  + x[ix,   jx+2] * (h[ix-2, jx+2] - h[ix+2, jx+2]) \
                                  + x[ix,   jx-2] * (h[ix+2, jx-2] - h[ix-2, jx-2]) \
                                  + x[ix+2, jx  ] * (h[ix+2, jx+2] - h[ix+2, jx-2]) \
                                  + x[ix-2, jx  ] * (h[ix-2, jx-2] - h[ix-2, jx+2]) \
                                  + x[ix+2, jx+2] * (h[ix,   jx+2] - h[ix+2, jx  ]) \
                                  + x[ix+2, jx-2] * (h[ix+2, jx  ] - h[ix,   jx-2]) \
                                  + x[ix-2, jx+2] * (h[ix-2, jx  ] - h[ix,   jx+2]) \
                                  + x[ix-2, jx-2] * (h[ix,   jx-2] - h[ix-2, jx  ]) \
                                ) / 48.
                    
                    bracket12 = ( \
                                  + (x[ix,   jx+2] - x[ix,   jx-2]) * (h[ix-1, jx  ] - h[ix+1, jx  ]) \
                                  + (x[ix+1, jx  ] - x[ix-1, jx  ]) * (h[ix,   jx+2] - h[ix,   jx-2]) \
                                  + x[ix,   jx+2] * (h[ix-1, jx+2] - h[ix+1, jx+2]) \
                                  + x[ix,   jx-2] * (h[ix+1, jx-2] - h[ix-1, jx-2]) \
                                  + x[ix+1, jx  ] * (h[ix+1, jx+2] - h[ix+1, jx-2]) \
                                  + x[ix-1, jx  ] * (h[ix-1, jx-2] - h[ix-1, jx+2]) \
                                  + x[ix+1, jx+2] * (h[ix,   jx+2] - h[ix+1, jx  ]) \
                                  + x[ix+1, jx-2] * (h[ix+1, jx  ] - h[ix,   jx-2]) \
                                  + x[ix-1, jx-2] * (h[ix,   jx-2] - h[ix-1, jx  ]) \
                                  + x[ix-1, jx+2] * (h[ix-1, jx  ] - h[ix,   jx+2]) \
                                ) / 24.
                    
                    bracket21 = ( \
                                  + (x[ix,   jx+1] - x[ix,   jx-1]) * (h[ix-2, jx  ] - h[ix+2, jx  ]) \
                                  + (x[ix+2, jx  ] - x[ix-2, jx  ]) * (h[ix,   jx+1] - h[ix,   jx-1]) \
                                  + x[ix,   jx+1] * (h[ix-2, jx+1] - h[ix+2, jx+1]) \
                                  + x[ix,   jx-1] * (h[ix+2, jx-1] - h[ix-2, jx-1]) \
                                  + x[ix+2, jx  ] * (h[ix+2, jx+1] - h[ix+2, jx-1]) \
                                  + x[ix-2, jx  ] * (h[ix-2, jx-1] - h[ix-2, jx+1]) \
                                  + x[ix+2, jx+1] * (h[ix,   jx+1] - h[ix+2, jx  ]) \
                                  + x[ix+2, jx-1] * (h[ix+2, jx  ] - h[ix,   jx-1]) \
                                  + x[ix-2, jx+1] * (h[ix-2, jx  ] - h[ix,   jx+1]) \
                                  + x[ix-2, jx-1] * (h[ix,   jx-1] - h[ix-2, jx  ]) \
                                ) / 24.
                    
                    bracket11 = ( \
                                  + (x[ix,   jx+1] - x[ix,   jx-1]) * (h[ix-1, jx  ] - h[ix+1, jx  ]) \
                                  + (x[ix+1, jx  ] - x[ix-1, jx  ]) * (h[ix,   jx+1] - h[ix,   jx-1]) \
                                  + x[ix,   jx+1] * (h[ix-1, jx+1] - h[ix+1, jx+1]) \
                                  + x[ix,   jx-1] * (h[ix+1, jx-1] - h[ix-1, jx-1]) \
                                  + x[ix-1, jx  ] * (h[ix-1, jx-1] - h[ix-1, jx+1]) \
                                  + x[ix+1, jx  ] * (h[ix+1, jx+1] - h[ix+1, jx-1]) \
                                  + x[ix+1, jx+1] * (h[ix,   jx+1] - h[ix+1, jx  ]) \
                                  + x[ix+1, jx-1] * (h[ix+1, jx  ] - h[ix,   jx-1]) \
                                  + x[ix-1, jx-1] * (h[ix,   jx-1] - h[ix-1, jx  ]) \
                                  + x[ix-1, jx+1] * (h[ix-1, jx  ] - h[ix,   jx+1]) \
                                ) / 12.
                    
                    y[iy, jy] += ( 25. * bracket11 - 10. * bracket12 - 10. * bracket21 + 4. * bracket22 ) * simpson_factor

