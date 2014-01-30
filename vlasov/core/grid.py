'''
Created on Mar 21, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import numpy as np


class Grid(object):
    '''
    Discrete grid in time, space, and velocity.
    '''


    def __init__(self, hdf5):
        '''
        Constructor
        '''
        
        assert hdf5 is not None
        
        
        self.hdf5  = hdf5
        
        self.tGrid = hdf5['t'][:,0,0]
        self.xGrid = hdf5['x'][:]
        self.vGrid = hdf5['v'][:]
        
        self.ht = self.tGrid[1] - self.tGrid[0]
        self.nt = len(self.tGrid)-1
        
        self.nx = len(self.xGrid)
        self.nv = len(self.vGrid)
        self.n  = self.nx * self.nv
        
        self.hx = self.xGrid[1] - self.xGrid[0]
        self.hv = self.vGrid[1] - self.vGrid[0]
        
        self.tMin = self.tGrid[ 1]
        self.tMax = self.tGrid[-1]
        self.xMin = self.xGrid[ 0]
        self.xMax = self.xGrid[-1]
        self.vMin = self.vGrid[ 0]
        self.vMax = self.vGrid[-1]
        
        
        print("")
        print("nt = %i (%i)" % (self.nt, len(self.tGrid)) )
        print("nx = %i" % (self.nx))
        print("nv = %i" % (self.nv))
        print
        print("ht = %f" % (self.ht))
        print("hx = %f" % (self.hx))
        print("hv = %f" % (self.hv))
        print("")
        print("tGrid:")
        print(self.tGrid)
        print("")
        print("xGrid:")
        print(self.xGrid)
        print("")
        print("vGrid:")
        print(self.vGrid)
        print("")
        
