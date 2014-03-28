'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np


cdef class PETScVlasovSolver(vlasov.solvers.preconditioner.TensorProductSciPy.PETScVlasovSolver):
    '''
    Implements a variational integrator with second order
    implicit midpoint time-derivative and Arakawa's J4
    discretisation of the Poisson brackets (J4=2J1-J2).
    '''
    
    cdef call_poisson_bracket(self, Vec F, Vec H, Vec Y, double factor):
        self.poisson_bracket.arakawa_J4(F, H, Y, factor)
