'''
Created on Mar 21, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import numpy as np


class Grid(object):
    '''
    Discrete grid in time, space, and velocity.
    '''


    def __init__(self, L=1.0,  nx=101,
                       vMin=-5.0, vMax=+5.0, nv=101, 
                       ht=0.01, nt=100, ntMax=0,
                       boundary_condition=None,             # None, 'dirichlet', 'neumann', 'weak'
                       hdf5_in=None, hdf5_out=None,
                       replay=False):
        '''
        Constructor
        '''
        
        self.hdf5  = hdf5_out
        self.bc    = boundary_condition
        self.ntMax = ntMax
        
        if hdf5_in != None:
#            self.tGrid = hdf5_in['grid']['t'][:]
#            self.xGrid = hdf5_in['grid']['x'][:]
#            self.vGrid = hdf5_in['grid']['v'][:]
            self.tGrid = hdf5_in['t'][:,0,0]
            self.xGrid = hdf5_in['x'][:]
            self.vGrid = hdf5_in['v'][:]
            
            if ntMax > 0 and len(self.tGrid) > ntMax+1:
                self.tGrid = self.tGrid[:ntMax+1]
            
            self.ht = self.tGrid[1] - self.tGrid[0]
            self.nt = len(self.tGrid)-1
            
            self.L = (self.xGrid[-1] - self.xGrid[0]) + (self.xGrid[1] - self.xGrid[0])
            
            if not replay:
                self.tGrid   += self.ht * self.nt
#                self.tGrid[0] = hdf5_in['grid']['t'][0]
                self.tGrid[0] = hdf5_in['t'][0]
            
        else:
            self.L    = L
            self.ht   = ht
            self.nt   = nt
            
            self.tGrid = ht * np.arange( 0, nt+1 )
            self.xGrid = np.linspace(  0.0,    L, nx, endpoint=False )
            self.vGrid = np.linspace( vMin, vMax, nv, endpoint=True  )
            
        
        if not replay:
            self.save_to_hdf5()
        
        
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
        
        
        print("nt = %i (%i)" % (self.nt, len(self.tGrid)) )
        print("nx = %i" % (self.nx))
        print("nv = %i" % (self.nv))
        print
        print("ht = %f" % (self.ht))
        print("hx = %f" % (self.hx))
        print("hv = %f" % (self.hv))
        print
        print("tGrid:")
        print(self.tGrid)
        print
        print("xGrid:")
        print(self.xGrid)
        print
        print("vGrid:")
        print(self.vGrid)
        print
        
    
    def append_time(self, timepoints):
        if self.nt < self.ntMax:
            ntCopy = len(timepoints)
            if self.nt + ntCopy > self.ntMax+1:
                ntCopy = self.ntMax - self.nt
            
            self.tGrid = np.concatenate([self.tGrid, timepoints[:ntCopy]])
            self.nt   += ntCopy
    

    def has_boundary_conditions(self):
        return self.isDirichlet() or self.isNeumann() or self.isWeak()
        
    def is_dirichlet(self):
        if self.bc == None:
            return False
        else:
            return self.bc.strip().lower() == 'dirichlet'
    
    def is_neumann(self):
        if self.bc == None:
            return False
        else:
            return self.bc.strip().lower() == 'neumann'
    
    def is_weak(self):
        if self.bc == None:
            return False
        else:
            return self.bc.strip().lower() == 'weak'
    
    
    def save_to_hdf5(self):
        if self.hdf5 == None:
            return
        
        if 'grid' in self.hdf5:
            hdf5_grid = self.hdf5['grid']
        else: 
            hdf5_grid = self.hdf5.create_group('grid')
            
        if 't' in hdf5_grid:
            hdf5_grid['t'][:] = self.tGrid
        else:
            hdf5_grid.create_dataset('t', data=self.tGrid)
        
        if 'x' in hdf5_grid:
            hdf5_grid['x'][:] = self.xGrid
        else:
            hdf5_grid.create_dataset('x', data=self.xGrid)
        
        if 'v' in hdf5_grid:
            hdf5_grid['v'][:] = self.vGrid
        else:
            hdf5_grid.create_dataset('v', data=self.vGrid)
        
