'''
Created on May 27, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from numpy cimport ndarray, float64_t
from petsc4py.PETSc cimport DMDA, Vec


cdef class VIDA(DMDA):
    
    
    def getGlobalArray(self, Vec tvec not None):
        cx, cdx = self.getCorners()
        dim     = self.getDim()
        dof     = self.getDof()
         
        shape = (cdx[0],)
         
        if dim > 1: shape += (cdx[1],)
        if dim > 2: shape += (cdx[2],)
        if dof > 1: shape += (dof,)
         
        if dof > 1: dim += 1
         
#         cdef ndarray[float64_t, ndim=dim] tarr = tvec.getArray().reshape(shape, order='f')
#         
#         return tarr
        
        return tvec.getArray().reshape(shape, order='f')
    
    
    def getLocalArray(self, Vec gvec not None, Vec lvec not None):
        self.globalToLocal(gvec, lvec)
         
        cx, cdx = self.getGhostCorners()
        dim     = self.getDim()
        dof     = self.getDof()
         
        shape = (cdx[0],)
         
        if dim > 1: shape += (cdx[1],)
        if dim > 2: shape += (cdx[2],)
        if dof > 1: shape += (dof,)
         
        if dof > 1: dim += 1
         
#         cdef ndarray[float64_t, ndim=dim] larr = lvec.getArray().reshape(shape, order='f')
#         
#         return larr
        
        return lvec.getArray().reshape(shape, order='f')
    