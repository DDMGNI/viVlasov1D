'''
Created on Mar 21, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from abc import ABCMeta, abstractmethod, abstractproperty


class VlasovSolver(): # Python 3: () -> (metaclass=ABCMeta) and remove __metaclass__ statement below
    '''
    Metaclass for Vlasov solvers.
    '''
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __init__(self, grid):
        self.grid  = grid
        self.nhist = 1
        
    
    @abstractmethod
    def initialise(self, f, h, p):
        pass
    
    
    @abstractmethod
    def solve(self, f, fhist, h):
        pass
    
