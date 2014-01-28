'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from vlasov.toolbox.Toolbox import Toolbox


cdef class PETScVlasovSolver(PETScVlasovSolverBase):
    '''
    Implements a variational integrator with first order
    finite-difference time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(self, Vec F, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        cdef double bracket, bracket11, bracket12, bracket21, bracket22
        cdef double coll_drag, coll_diff
        cdef double collisions     = 0.
        cdef double regularisation = 0.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double[:,:] fd    = self.da1.getLocalArray(F, self.localFd)
        cdef double[:,:] y     = self.da1.getGlobalArray(Y)
        cdef double[:,:] h_ave = self.da1.getLocalArray(self.Have, self.localHave)
        
        cdef double[:] v = self.grid.v
        cdef double[:] u = self.Up.getArray()
        cdef double[:] a = self.Ap.getArray()
        
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy] = fd[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    bracket22 = ( \
                                  + (fd[ix,   jx+2] - fd[ix,   jx-2]) * (h_ave[ix-2, jx  ] - h_ave[ix+2, jx  ]) \
                                  + (fd[ix+2, jx  ] - fd[ix-2, jx  ]) * (h_ave[ix,   jx+2] - h_ave[ix,   jx-2]) \
                                  + fd[ix,   jx+2] * (h_ave[ix-2, jx+2] - h_ave[ix+2, jx+2]) \
                                  + fd[ix,   jx-2] * (h_ave[ix+2, jx-2] - h_ave[ix-2, jx-2]) \
                                  + fd[ix+2, jx  ] * (h_ave[ix+2, jx+2] - h_ave[ix+2, jx-2]) \
                                  + fd[ix-2, jx  ] * (h_ave[ix-2, jx-2] - h_ave[ix-2, jx+2]) \
                                  + fd[ix+2, jx+2] * (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) \
                                  + fd[ix+2, jx-2] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2]) \
                                  + fd[ix-2, jx+2] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx+2]) \
                                  + fd[ix-2, jx-2] * (h_ave[ix,   jx-2] - h_ave[ix-2, jx  ]) \
                                ) / 48.
                    
                    bracket12 = ( \
                                  + (fd[ix,   jx+2] - fd[ix,   jx-2]) * (h_ave[ix-1, jx  ] - h_ave[ix+1, jx  ]) \
                                  + (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (h_ave[ix,   jx+2] - h_ave[ix,   jx-2]) \
                                  + fd[ix,   jx+2] * (h_ave[ix-1, jx+2] - h_ave[ix+1, jx+2]) \
                                  + fd[ix,   jx-2] * (h_ave[ix+1, jx-2] - h_ave[ix-1, jx-2]) \
                                  + fd[ix+1, jx  ] * (h_ave[ix+1, jx+2] - h_ave[ix+1, jx-2]) \
                                  + fd[ix-1, jx  ] * (h_ave[ix-1, jx-2] - h_ave[ix-1, jx+2]) \
                                  + fd[ix+1, jx+2] * (h_ave[ix,   jx+2] - h_ave[ix+1, jx  ]) \
                                  + fd[ix+1, jx-2] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-2]) \
                                  + fd[ix-1, jx-2] * (h_ave[ix,   jx-2] - h_ave[ix-1, jx  ]) \
                                  + fd[ix-1, jx+2] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx+2]) \
                                ) / 24.
                    
                    bracket21 = ( \
                                  + (fd[ix,   jx+1] - fd[ix,   jx-1]) * (h_ave[ix-2, jx  ] - h_ave[ix+2, jx  ]) \
                                  + (fd[ix+2, jx  ] - fd[ix-2, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                                  + fd[ix,   jx+1] * (h_ave[ix-2, jx+1] - h_ave[ix+2, jx+1]) \
                                  + fd[ix,   jx-1] * (h_ave[ix+2, jx-1] - h_ave[ix-2, jx-1]) \
                                  + fd[ix+2, jx  ] * (h_ave[ix+2, jx+1] - h_ave[ix+2, jx-1]) \
                                  + fd[ix-2, jx  ] * (h_ave[ix-2, jx-1] - h_ave[ix-2, jx+1]) \
                                  + fd[ix+2, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+2, jx  ]) \
                                  + fd[ix+2, jx-1] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-1]) \
                                  + fd[ix-2, jx+1] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx+1]) \
                                  + fd[ix-2, jx-1] * (h_ave[ix,   jx-1] - h_ave[ix-2, jx  ]) \
                                ) / 24.
                    
                    bracket11 = ( \
                                  + (fd[ix,   jx+1] - fd[ix,   jx-1]) * (h_ave[ix-1, jx  ] - h_ave[ix+1, jx  ]) \
                                  + (fd[ix+1, jx  ] - fd[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                                  + fd[ix,   jx+1] * (h_ave[ix-1, jx+1] - h_ave[ix+1, jx+1]) \
                                  + fd[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) \
                                  + fd[ix-1, jx  ] * (h_ave[ix-1, jx-1] - h_ave[ix-1, jx+1]) \
                                  + fd[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                                  + fd[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                                  + fd[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1]) \
                                  + fd[ix-1, jx-1] * (h_ave[ix,   jx-1] - h_ave[ix-1, jx  ]) \
                                  + fd[ix-1, jx+1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx+1]) \
                                ) / 12.
                    
                    bracket = 0.5 * ( 25. * bracket11 - 10. * bracket12 - 10. * bracket21 + 4. * bracket22 ) / 9. * self.grid.hx_inv * self.grid.hv_inv
                    
                    
                    # collision operator
                    if self.nu > 0.:
                        coll_drag = ( (v[j+1] - u[ix]) * fd[ix, jx+1] - (v[j-1] - u[ix]) * fd[ix, jx-1] ) * a[ix]
                        
                        coll_diff = ( fd[ix, jx+1] - 2. * fd[ix, jx] + fd[ix, jx-1] )
                        
                        collisions = \
                              - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                              - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv \
                    
                    # regularisation
                    if self.regularisation != 0.:
                        regularisation = \
                              + self.grid.ht * self.regularisation * self.grid.hx2_inv * ( 2. * fd[ix, jx] - fd[ix+1, jx] - fd[ix-1, jx] ) \
                              + self.grid.ht * self.regularisation * self.grid.hv2_inv * ( 2. * fd[ix, jx] - fd[ix, jx+1] - fd[ix, jx-1] )
                    
                    # solution
                    y[iy, jy] = fd[ix, jx] * self.grid.ht_inv \
                              + bracket \
                              + collisions \
                              + regularisation
    
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def function(self, Vec F, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        cdef double bracket, bracket11, bracket12, bracket21, bracket22
        cdef double coll_drag, coll_diff
        cdef double collisions     = 0.
        cdef double regularisation = 0.
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.Fave.set(0.)
        self.Fave.axpy(0.5, self.Fh)
        self.Fave.axpy(0.5, F)
        
        cdef double[:,:] y     = self.da1.getGlobalArray(Y)
        cdef double[:,:] fp    = self.da1.getLocalArray(F, self.localFp)
        cdef double[:,:] fh    = self.da1.getLocalArray(self.Fh, self.localFh)
        cdef double[:,:] f_ave = self.da1.getLocalArray(self.Fave, self.localFave)
        cdef double[:,:] h_ave = self.da1.getLocalArray(self.Have, self.localHave)
        
        cdef double[:] v  = self.grid.v
        cdef double[:] up = self.Up.getArray()
        cdef double[:] uh = self.Uh.getArray()
        cdef double[:] ap = self.Ap.getArray()
        cdef double[:] ah = self.Ah.getArray()
        
        
        for j in range(ys, ye):
            jx = j-ys+self.grid.stencil
            jy = j-ys

            if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                # Dirichlet Boundary Conditions
                y[0:xe-xs, jy] = fp[self.grid.stencil:xe-xs+self.grid.stencil, jx]
                
            else:
                # Vlasov equation
                for i in range(xs, xe):
                    ix = i-xs+self.grid.stencil
                    iy = i-xs
            
                    bracket22 = ( \
                                  + (f_ave[ix,   jx+2] - f_ave[ix,   jx-2]) * (h_ave[ix-2, jx  ] - h_ave[ix+2, jx  ]) \
                                  + (f_ave[ix+2, jx  ] - f_ave[ix-2, jx  ]) * (h_ave[ix,   jx+2] - h_ave[ix,   jx-2]) \
                                  + f_ave[ix,   jx+2] * (h_ave[ix-2, jx+2] - h_ave[ix+2, jx+2]) \
                                  + f_ave[ix,   jx-2] * (h_ave[ix+2, jx-2] - h_ave[ix-2, jx-2]) \
                                  + f_ave[ix+2, jx  ] * (h_ave[ix+2, jx+2] - h_ave[ix+2, jx-2]) \
                                  + f_ave[ix-2, jx  ] * (h_ave[ix-2, jx-2] - h_ave[ix-2, jx+2]) \
                                  + f_ave[ix+2, jx+2] * (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) \
                                  + f_ave[ix+2, jx-2] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2]) \
                                  + f_ave[ix-2, jx+2] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx+2]) \
                                  + f_ave[ix-2, jx-2] * (h_ave[ix,   jx-2] - h_ave[ix-2, jx  ]) \
                                ) / 48.
                    
                    bracket12 = ( \
                                  + (f_ave[ix,   jx+2] - f_ave[ix,   jx-2]) * (h_ave[ix-1, jx  ] - h_ave[ix+1, jx  ]) \
                                  + (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ]) * (h_ave[ix,   jx+2] - h_ave[ix,   jx-2]) \
                                  + f_ave[ix,   jx+2] * (h_ave[ix-1, jx+2] - h_ave[ix+1, jx+2]) \
                                  + f_ave[ix,   jx-2] * (h_ave[ix+1, jx-2] - h_ave[ix-1, jx-2]) \
                                  + f_ave[ix+1, jx  ] * (h_ave[ix+1, jx+2] - h_ave[ix+1, jx-2]) \
                                  + f_ave[ix-1, jx  ] * (h_ave[ix-1, jx-2] - h_ave[ix-1, jx+2]) \
                                  + f_ave[ix+1, jx+2] * (h_ave[ix,   jx+2] - h_ave[ix+1, jx  ]) \
                                  + f_ave[ix+1, jx-2] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-2]) \
                                  + f_ave[ix-1, jx-2] * (h_ave[ix,   jx-2] - h_ave[ix-1, jx  ]) \
                                  + f_ave[ix-1, jx+2] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx+2]) \
                                ) / 24.
                    
                    bracket21 = ( \
                                  + (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) * (h_ave[ix-2, jx  ] - h_ave[ix+2, jx  ]) \
                                  + (f_ave[ix+2, jx  ] - f_ave[ix-2, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                                  + f_ave[ix,   jx+1] * (h_ave[ix-2, jx+1] - h_ave[ix+2, jx+1]) \
                                  + f_ave[ix,   jx-1] * (h_ave[ix+2, jx-1] - h_ave[ix-2, jx-1]) \
                                  + f_ave[ix+2, jx  ] * (h_ave[ix+2, jx+1] - h_ave[ix+2, jx-1]) \
                                  + f_ave[ix-2, jx  ] * (h_ave[ix-2, jx-1] - h_ave[ix-2, jx+1]) \
                                  + f_ave[ix+2, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+2, jx  ]) \
                                  + f_ave[ix+2, jx-1] * (h_ave[ix+2, jx  ] - h_ave[ix,   jx-1]) \
                                  + f_ave[ix-2, jx+1] * (h_ave[ix-2, jx  ] - h_ave[ix,   jx+1]) \
                                  + f_ave[ix-2, jx-1] * (h_ave[ix,   jx-1] - h_ave[ix-2, jx  ]) \
                                ) / 24.
                    
                    bracket11 = ( \
                                  + (f_ave[ix,   jx+1] - f_ave[ix,   jx-1]) * (h_ave[ix-1, jx  ] - h_ave[ix+1, jx  ]) \
                                  + (f_ave[ix+1, jx  ] - f_ave[ix-1, jx  ]) * (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) \
                                  + f_ave[ix,   jx+1] * (h_ave[ix-1, jx+1] - h_ave[ix+1, jx+1]) \
                                  + f_ave[ix,   jx-1] * (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) \
                                  + f_ave[ix-1, jx  ] * (h_ave[ix-1, jx-1] - h_ave[ix-1, jx+1]) \
                                  + f_ave[ix+1, jx  ] * (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) \
                                  + f_ave[ix+1, jx+1] * (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) \
                                  + f_ave[ix+1, jx-1] * (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1]) \
                                  + f_ave[ix-1, jx-1] * (h_ave[ix,   jx-1] - h_ave[ix-1, jx  ]) \
                                  + f_ave[ix-1, jx+1] * (h_ave[ix-1, jx  ] - h_ave[ix,   jx+1]) \
                                ) / 12.
                    
                    bracket = ( 25. * bracket11 - 10. * bracket12 - 10. * bracket21 + 4. * bracket22 ) / 9. * self.grid.hx_inv * self.grid.hv_inv
                    
                    
                    # collision operator
                    if self.nu > 0.:
                        coll_drag = ( (v[j+1] - up[ix]) * fp[ix, jx+1] - (v[j-1] - up[ix]) * fp[ix, jx-1] ) * ap[ix] \
                                  + ( (v[j+1] - uh[ix]) * fh[ix, jx+1] - (v[j-1] - uh[ix]) * fh[ix, jx-1] ) * ah[ix]
                        
                        coll_diff = ( fp[ix, jx+1] - 2. * fp[ix, jx] + fp[ix, jx-1] ) \
                                  + ( fh[ix, jx+1] - 2. * fh[ix, jx] + fh[ix, jx-1] )
                        
                        collisions = \
                                   - 0.5 * self.nu * self.coll_drag * coll_drag * self.grid.hv_inv * 0.5 \
                                   - 0.5 * self.nu * self.coll_diff * coll_diff * self.grid.hv2_inv \
                    
                    # regularisation
                    if self.regularisation != 0.:
                        regularisation = \
                                       + self.grid.ht * self.regularisation * self.grid.hx2_inv * ( 2. * fp[ix, jx] - fp[ix+1, jx] - fp[ix-1, jx] ) \
                                       + self.grid.ht * self.regularisation * self.grid.hv2_inv * ( 2. * fp[ix, jx] - fp[ix, jx+1] - fp[ix, jx-1] )
                    
                    # solution
                    y[iy, jy] = (fp[ix, jx] - fh[ix, jx]) * self.grid.ht_inv \
                              + bracket \
                              + collisions \
                              + regularisation
