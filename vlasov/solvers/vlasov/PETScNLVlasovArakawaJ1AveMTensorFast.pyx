'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np


cdef class PETScVlasovSolver(vlasov.solvers.preconditioner.TensorProductFast.PETScVlasovSolver):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    cdef formBandedPreconditionerMatrix(self, dcomplex[:,:] matrix, dcomplex eigen):
        cdef int j
        
        cdef double[:] v = self.grid.v
        
        cdef double arak_fac_J1 = 0.5 / (12. * self.grid.hx * self.grid.hv)
        
        
        cdef dcomplex[:] diagm = np.zeros(self.grid.nv, dtype=np.cdouble)
        cdef dcomplex[:] diag  = np.ones (self.grid.nv, dtype=np.cdouble)
        cdef dcomplex[:] diagp = np.zeros(self.grid.nv, dtype=np.cdouble)
        
        for j in range(2, self.grid.nv-2):
            diagm[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] - self.grid.hv2 ) * arak_fac_J1 + 0.25 * self.grid.ht_inv
            diag [j] = eigen *         4. * self.grid.hv * v[j]                   * arak_fac_J1 + 0.50 * self.grid.ht_inv
            diagp[j] = eigen * 0.5 * ( 2. * self.grid.hv * v[j] + self.grid.hv2 ) * arak_fac_J1 + 0.25 * self.grid.ht_inv
        
        matrix[1, 1:  ] = diagp[:-1]
        matrix[2,  :  ] = diag [:]
        matrix[3,  :-1] = diagm[1:]
        
    
    cdef call_poisson_bracket(self, Vec F, Vec H, Vec Y, double factor):
        self.poisson_bracket.arakawa_J1(F, H, Y, factor)
        
    cdef call_time_derivative(self, Vec F, Vec Y):
        self.time_derivative.midpoint(F, Y)
