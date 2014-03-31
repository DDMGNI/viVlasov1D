'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc


cdef class PETScVlasovSolver(PETScVlasovSolverBase):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def __init__(self,
                 VIDA da1  not None,
                 Grid grid not None,
                 Vec H0  not None,
                 Vec H1p not None,
                 Vec H1h not None,
                 Vec H2p not None,
                 Vec H2h not None,
                 double charge=-1.,
                 double coll_freq=0.,
                 double coll_diff=1.,
                 double coll_drag=1.,
                 double regularisation=0.):
        '''
        Constructor
        '''
        
        super().__init__(da1, grid, H0, H1p, H1h, H2p, H2h, charge, 0., 0., 0., 0.)
        
        # collision frequency
        self.coll_freq = coll_freq

        # create vectors
        self.bracket = self.da1.createGlobalVec()
        
        
        
    cdef jacobian_solver(self, Vec F, Vec Y):
        Y.set(0.)
        
        self.bracket.set(0.)
        self.call_poisson_bracket(self.Fave, self.Have, self.bracket, 1.0)
        self.call_poisson_bracket(F, self.bracket, Y, 0.5)
        
        self.bracket.set(0.)
        self.call_poisson_bracket(F, self.Have, self.bracket, 0.5)
        self.call_poisson_bracket(self.Fave, self.bracket, Y, self.coll_freq)
        
        Y.axpy(1.0, self.bracket)
        self.call_time_derivative(F, Y)
    
    
    cdef function_solver(self, Vec F, Vec Y):
        self.Fave.set(0.)
        self.Fave.axpy(0.5, self.Fh)
        self.Fave.axpy(0.5, F)
        
        self.Fder.set(0.)
        self.Fder.axpy(+1, F)
        self.Fder.axpy(-1, self.Fh)
        
        Y.set(0.)
        
        self.bracket.set(0.)
        self.call_poisson_bracket(self.Fave, self.Have, self.bracket, 1.0)
        self.call_poisson_bracket(self.Fave, self.bracket, Y, self.coll_freq)
        
        Y.axpy(1.0, self.bracket)
        self.call_time_derivative(self.Fder, Y)
        
    
    cdef call_poisson_bracket(self, Vec F, Vec H, Vec Y, double factor):
        self.poisson_bracket.arakawa_J1(F, H, Y, factor)
        
