'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np


cdef class PETScVlasovSolver(vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4TensorFast.PETScVlasovSolver):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    cdef jacobianSolver(self, Vec F, Vec Y):
        Y.set(0.)
        self.poisson_bracket.arakawa_J1(F, self.Have, Y, 0.5)
        self.time_derivative.time_derivative(F, Y)
        self.collisions.collT(F, Y, self.Np, self.Up, self.Ep, self.Ap, 0.5)
        self.regularisation.regularisation(F, Y, 1.0)
    
    
    cdef functionSolver(self, Vec F, Vec Y):
        self.Fave.set(0.)
        self.Fave.axpy(0.5, self.Fh)
        self.Fave.axpy(0.5, F)
        
        self.Fder.set(0.)
        self.Fder.axpy(+1, F)
        self.Fder.axpy(-1, self.Fh)
        
        Y.set(0.)
        
        self.poisson_bracket.arakawa_J1(self.Fave, self.Have, Y, 1.0)
        self.time_derivative.time_derivative(self.Fder, Y)
        self.collisions.collT(F, Y, self.Np, self.Up, self.Ep, self.Ap, 0.5)
        self.collisions.collT(F, Y, self.Nh, self.Uh, self.Eh, self.Ah, 0.5)
#         self.regularisation.regularisation(F, Y, 1.0)
        
