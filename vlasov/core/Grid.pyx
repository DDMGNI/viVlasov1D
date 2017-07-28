'''
Created on 16.01.2014

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import numpy as np

cdef class Grid(object):
    '''
    The grid class bundles all information related to the (x,v) grid.
    '''

    def create(self, 
                 double[:] x,
                 double[:] v,
                 int    nt, int    nx, int    nv,
                 double ht, double hx, double hv,
                 int stencil):
        '''
        Constructor
        '''
        
        self.stencil = stencil
        
        self.t  = np.linspace(0, ht*nt, nt+1)
        self.x  = x
        self.v  = v
        
        self.v2 = np.power(v, 2)
        
        self.nt = nt
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv

        self.ht_inv  = 1. / self.ht 
        self.hx_inv  = 1. / self.hx 
        self.hv_inv  = 1. / self.hv
         
        self.hx2     = self.hx**2
        self.hv2     = self.hv**2
        
        self.hx2_inv = 1. / self.hx2 
        self.hv2_inv = 1. / self.hv2
        
        return self
        
    
    def load_from_hdf5(self, hdf5):
        
        assert hdf5 is not None
        
        self.t = hdf5['t'][:,0,0]
        self.x = hdf5['x'][:]
        self.v = hdf5['v'][:]
        
        self.v2 = np.power(self.v, 2)
        
        self.nt = len(self.t)-1
        self.nx = len(self.x)
        self.nv = len(self.v)
        
        self.ht = self.t[1] - self.t[0]
        self.hx = self.x[1] - self.x[0]
        self.hv = self.v[1] - self.v[0]
        
        self.ht_inv  = 1. / self.ht 
        self.hx_inv  = 1. / self.hx 
        self.hv_inv  = 1. / self.hv
         
        self.hx2     = self.hx**2
        self.hv2     = self.hv**2
        
        self.hx2_inv = 1. / self.hx2 
        self.hv2_inv = 1. / self.hv2 
        
#         self.tMin = self.t[ 1]
#         self.tMax = self.t[-1]
#         self.xMin = self.x[ 0]
#         self.xMax = self.x[-1]
#         self.vMin = self.v[ 0]
#         self.vMax = self.v[-1]
        
        
        print("")
        print("nt = %i (%i)" % (self.nt, len(self.t)) )
        print("nx = %i" % (self.nx))
        print("nv = %i" % (self.nv))
        print
        print("ht = %f" % (self.ht))
        print("hx = %f" % (self.hx))
        print("hv = %f" % (self.hv))
        print("")
        print("tGrid:")
        print(self.t)
        print("")
        print("xGrid:")
        print(self.x)
        print("")
        print("vGrid:")
        print(self.v)
        print("")
        
        return self
        
        

    def tMin(self):
        return self.t[0]
    
    def tMax(self):
        return self.t[-1]

    def xMin(self):
        return self.x[0]
    
    def xMax(self):
        return self.x[-1]

    def vMin(self):
        return self.v[0]
    
    def vMax(self):
        return self.v[-1]

    def xLength(self):
        return self.xMax() - self.xMin() + self.hx
    
    def vLength(self):
        return self.vMax() - self.vMin()
    
