'''
Created on 16.01.2014

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cdef class Grid(object):
    '''
    The grid class bundles all information related to the (x,v) grid.
    '''

    def __init__(self, 
                 np.ndarray[np.float64_t, ndim=1] x,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t  nt, np.uint64_t  nx, np.uint64_t  nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv):
        '''
        Constructor
        '''
        
        self.x = x
        self.v = v
        
        self.nt = nt
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv

        self.ht_inv  = 1. / self.ht 
        self.hx_inv  = 1. / self.hx 
        self.hv_inv  = 1. / self.hv
         
        self.hx2     = hx**2
        self.hv2     = hv**2
        
        self.hx2_inv = 1. / self.hx2 
        self.hv2_inv = 1. / self.hv2 
        

    def xMin(self):
        return self.x[0]
    
    def xMax(self):
        return self.x[-1]

    def vMin(self):
        return self.v[0]
    
    def vMax(self):
        return self.v[-1]

    def xLength(self):
        return self.xMax() - self.xMin()
    
    def vLength(self):
        return self.vMax() - self.vMin()
    
