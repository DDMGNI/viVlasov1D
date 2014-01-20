'''
Created on May 24, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport Vec

from vlasov.toolbox.Arakawa import Arakawa


cdef class PETScExplicitSolver(object):
    '''
    PETSc/Cython Base Class of an Explicit Vlasov-Poisson Solver
    '''
    
    
    def __init__(self, 
                 VIDA da1  not None,
                 Grid grid not None,
                 Vec H0    not None,
                 Vec H1    not None,
                 Vec H2    not None):
        '''
        Constructor
        '''
        
        # distributed array and grid
        self.da1  = da1
        self.grid = grid
        
        # Hamiltonians
        self.H0 = H0
        self.H1 = H1
        self.H2 = H2
        
        # create global vectors
        self.X1 = self.da1.createGlobalVec()
        self.X2 = self.da1.createGlobalVec()
        self.X3 = self.da1.createGlobalVec()
        self.X4 = self.da1.createGlobalVec()
        
        # create local vectors
        self.localH0   = da1.createLocalVec()
        self.localH1   = da1.createLocalVec()
        self.localH2   = da1.createLocalVec()
        
        # create toolbox object
        self.arakawa = Arakawa(da1, grid)
    
    
