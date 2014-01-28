'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc


cdef class PETScVlasovSolverKinetic(vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4.PETScVlasovSolver):
    '''
    Implements a variational integrator with first order
    finite-difference time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    def update_previous(self, Vec F):
        F.copy(self.Fp)
        
        self.H0.copy(self.Have)
