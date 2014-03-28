'''
# cython: profile=True
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
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def formJacobian(self, Mat A):
        cdef int i, j, ix, jx
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef double[:,:] h_ave = self.da1.getLocalArray(self.Have, self.localHave)
        
        cdef double[:] v  = self.grid.v
        cdef double[:] up = self.Up.getArray()
        cdef double[:] ap = self.Ap.getArray()
        
        
#         cdef double time_fac      = 0.
#         cdef double arak_fac_J1   = 0.
#         cdef double arak_fac_J2   = 0.
#         cdef double coll_drag_fac = 0.
#         cdef double coll_diff_fac = 0.
        
        cdef double time_fac      = self.grid.ht_inv / 36.
        cdef double arak_fac_J1   = + 1.0 / (12. * self.grid.hx * self.grid.hv)
        cdef double arak_fac_J2   = - 0.5 / (24. * self.grid.hx * self.grid.hv)
        
        cdef double coll_drag_fac = - 0.5 * self.nu * self.coll_drag * self.grid.hv_inv * 0.5
        cdef double coll_diff_fac = - 0.5 * self.nu * self.coll_diff * self.grid.hv2_inv
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        row.field = 0
        col.field = 0
        
        
        # Vlasov Equation
        for i in range(xs, xe):
            ix = i-xs+self.grid.stencil
            
            for j in range(ys, ye):
                jx = j-ys+self.grid.stencil

                row.index = (i,j)
                
                # Dirichlet boundary conditions
                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    A.setValueStencil(row, row, 1.0)
                    
                else:
                    for index, value in [
                            ((i-2, j  ), - (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
                            ((i-1, j-1), + 1. * time_fac \
                                         - (h_ave[ix-1, jx  ] - h_ave[ix,   jx-1]) * arak_fac_J1 \
                                         - (h_ave[ix-2, jx  ] - h_ave[ix,   jx-2]) * arak_fac_J2 \
                                         - (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J2),
                            ((i-1, j  ), + 4. * time_fac \
                                         - (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) * arak_fac_J1 \
                                         - (h_ave[ix-1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J1 \
                                         - self.grid.ht * self.regularisation * self.grid.hx2_inv),
                            ((i-1, j+1), + 1. * time_fac \
                                         - (h_ave[ix,   jx+1] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
                                         - (h_ave[ix,   jx+2] - h_ave[ix-2, jx  ]) * arak_fac_J2 \
                                         - (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
                            ((i,   j-2), + (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
                            ((i,   j-1), + 4. * time_fac \
                                         + (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
                                         + (h_ave[ix+1, jx-1] - h_ave[ix-1, jx-1]) * arak_fac_J1 \
                                         - coll_drag_fac * ( v[j-1] - up[ix  ] ) * ap[ix  ] \
                                         + coll_diff_fac \
                                         - self.grid.ht * self.regularisation * self.grid.hv2_inv),
                            ((i,   j  ), +16. * time_fac \
                                         - 2. * coll_diff_fac \
                                         + 2. * self.grid.ht * self.regularisation * (self.grid.hx2_inv + self.grid.hv2_inv)),
                            ((i,   j+1), + 4. * time_fac \
                                         - (h_ave[ix+1, jx  ] - h_ave[ix-1, jx  ]) * arak_fac_J1 \
                                         - (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) * arak_fac_J1 \
                                         + coll_drag_fac * ( v[j+1] - up[ix  ] ) * ap[ix  ] \
                                         + coll_diff_fac \
                                         - self.grid.ht * self.regularisation * self.grid.hv2_inv),
                            ((i,   j+2), - (h_ave[ix+1, jx+1] - h_ave[ix-1, jx+1]) * arak_fac_J2),
                            ((i+1, j-1), + 1. * time_fac \
                                         + (h_ave[ix+1, jx  ] - h_ave[ix,   jx-1]) * arak_fac_J1 \
                                         + (h_ave[ix+2, jx  ] - h_ave[ix,   jx-2]) * arak_fac_J2 \
                                         + (h_ave[ix+1, jx+1] - h_ave[ix-1, jx-1]) * arak_fac_J2),
                            ((i+1, j  ), + 4. * time_fac \
                                         + (h_ave[ix,   jx+1] - h_ave[ix,   jx-1]) * arak_fac_J1 \
                                         + (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J1 \
                                         - self.grid.ht * self.regularisation * self.grid.hx2_inv),
                            ((i+1, j+1), + 1. * time_fac \
                                         + (h_ave[ix,   jx+1] - h_ave[ix+1, jx  ]) * arak_fac_J1 \
                                         + (h_ave[ix,   jx+2] - h_ave[ix+2, jx  ]) * arak_fac_J2 \
                                         + (h_ave[ix-1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J2),
                            ((i+2, j  ), + (h_ave[ix+1, jx+1] - h_ave[ix+1, jx-1]) * arak_fac_J2),
                        ]:

                        col.index = index
                        A.setValueStencil(row, col, value)
                        
        
        A.assemble()



    cdef call_poisson_bracket(self, Vec F, Vec H, Vec Y, double factor):
        self.poisson_bracket.arakawa_J4(F, H, Y, factor)
        
    cdef call_time_derivative(self, Vec F, Vec Y):
        self.time_derivative.simpson(F, Y)
