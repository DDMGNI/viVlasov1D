'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc


cdef class DoubleBracket(object):
    '''
    
    '''
    
    def __init__(self,
                 VIDA da1  not None,
                 Grid grid not None,
                 PoissonBracket poisson_bracket not None,
                 double coll_freq=0.):
        '''
        Constructor
        '''
        
        # distributed arrays and grid
        self.da1  = da1
        self.grid = grid
        
        # poisson bracket
        self.poisson_bracket = poisson_bracket
        
        # collision parameters
        self.coll_freq = coll_freq
        
        # create local vectors
        self.bracket = da1.createGlobalVec()
        

    @staticmethod
    def create(str  type not None,
               VIDA da1  not None,
               Grid grid not None,
               PoissonBracket poisson_bracket not None,
               double coll_freq=0.):
        
        
        if type == 'ffh':
            return DoubleBracketFFH(da1, grid, poisson_bracket, coll_freq)
        elif type == 'fhh':
            return DoubleBracketFHH(da1, grid, poisson_bracket, coll_freq)
        else:
            return DoubleBracket(da1, grid, poisson_bracket, coll_freq)
        
    
    cdef void jacobian(self, Vec F, Vec Fave, Vec Have, Vec Y, double factor):
        pass
    
    cdef void function(self, Vec Fave, Vec Have, Vec Y, double factor):
        pass
    
    
    
cdef class DoubleBracketFFH(DoubleBracket):

    cdef void jacobian(self, Vec F, Vec Fave, Vec Have, Vec Y, double factor):
        self.bracket.set(0.)
        self.poisson_bracket.function(Fave, Have, self.bracket, 1.0)
        self.poisson_bracket.function(F, self.bracket, Y, self.coll_freq * factor)
        
        self.bracket.set(0.)
        self.poisson_bracket.function(F, Have, self.bracket, 1.0)
        self.poisson_bracket.function(Fave, self.bracket, Y, self.coll_freq * factor)
        
        
    cdef void function(self, Vec Fave, Vec Have, Vec Y, double factor):
        self.bracket.set(0.)
        self.poisson_bracket.function(Fave, Have, self.bracket, 1.0)
        self.poisson_bracket.function(Fave, self.bracket, Y, self.coll_freq * factor)
        
    

cdef class DoubleBracketFHH(DoubleBracket):
    
    cdef void jacobian(self, Vec F, Vec Fave, Vec Have, Vec Y, double factor):
        self.bracket.set(0.)
        self.poisson_bracket.function(F, Have, self.bracket, 1.0)
        self.poisson_bracket.function(self.bracket, Have, Y, self.coll_freq * factor)
        
        
    cdef void function(self, Vec Fave, Vec Have, Vec Y, double factor):
        self.bracket.set(0.)
        self.poisson_bracket.function(Fave, Have, self.bracket, 1.0)
        self.poisson_bracket.function(self.bracket, Have, Y, self.coll_freq * factor)
        
    
