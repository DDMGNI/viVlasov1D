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
    
    def formJacobian(self, Mat J):
        J.zeroEntries()
        
        self.time_derivative.jacobian(J)
        self.poisson_bracket.jacobian(J, self.Hh, 0.5)
        self.collision_operator.jacobian(J, self.Np, self.Up, self.Ep, self.Ap, 0.5)
        self.regularisation.jacobian(J, 1.0)
        
        J.assemble()
        
    
    cdef jacobian_solver(self, Vec F, Vec Y):
        Y.set(0.)
        
        self.time_derivative.function(F, Y)
        self.poisson_bracket.function(F, self.Hh, Y, 0.5)
        self.double_bracket.jacobian(F, self.Fp, self.Hh, Y, 0.5)
        self.collision_operator.function(F, Y, self.Np, self.Up, self.Ep, self.Ap, 0.5)
        self.regularisation.function(F, Y, 1.0)
        
    
    cdef function_solver(self, Vec F, Vec Y):
        self.Fave.set(0.)
        self.Fave.axpy(0.5, self.Fh)
        self.Fave.axpy(0.5, F)
        
        self.Fder.set(0.)
        self.Fder.axpy(+1, F)
        self.Fder.axpy(-1, self.Fh)
        
        Y.set(0.)
        
        self.time_derivative.function(self.Fder, Y)
        self.poisson_bracket.function(F,       self.Hh, Y, 0.5)
        self.poisson_bracket.function(self.Fh, self.Hp, Y, 0.5)
        self.double_bracket.function(F,       self.Hh, Y, 0.5)
        self.double_bracket.function(self.Fh, self.Hp, Y, 0.5)
        self.collision_operator.function(F,       Y, self.Np, self.Up, self.Ep, self.Ap, 0.5)
        self.collision_operator.function(self.Fh, Y, self.Nh, self.Uh, self.Eh, self.Ah, 0.5)
        
        
