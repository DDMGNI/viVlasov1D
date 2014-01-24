'''
Created on May 27, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from numpy cimport ndarray, float64_t
from numpy  import swapaxes
from petsc4py.PETSc cimport DMDA, Vec


cdef class VIDA(DMDA):
    
    cdef reshape(self, ndarray vec, bool local):
        if local:
            cstarts, csizes = self.getGhostCorners()
        else:
            cstarts, csizes = self.getCorners()
        
#         xm, ym, zm = csizes
        
        dim = self.getDim()
        dof = self.getDof()
         
        shape   = (csizes[0],)
#         strides = (dof,)[:dim]
         
        if dim > 1: shape += (csizes[1],)
        if dim > 2: shape += (csizes[2],)
#         if dof > 1: shape += (dof,)
        if dof > 1: shape = (dof,) + shape

#         if dim > 1: strides += (dof*xm,)
#         if dim > 2: strides += (dof*xm*ym,)
#         if dof > 1: strides += (1,)

#         print (dim, dof, shape, strides)
        
#         if dof == 2:
#             shape1 = (dof,) + shape
#             shape2 = shape + (dof,)
#             print(shape1,shape2)
#             tvec = vec.reshape(shape1, order='f')
#             fvec = empty(shape2)
#             fvec[:,:,0] = tvec[0,:,:] 
#             fvec[:,:,1] = tvec[1,:,:] 
#             
#             return fvec
#         else:
        
        arr = vec.reshape(shape, order='f')
        
        if dof > 1:
            arr = swapaxes(arr, 0, 1)
            if dim > 1: arr = swapaxes(arr, 1, 2)
            if dim > 2: arr = swapaxes(arr, 2, 3)
        
        return arr 
#         return vec.reshape(shape, order='f')
    
    
    cpdef getGlobalArray(self, Vec tvec):
        return self.reshape(tvec.getArray(), local=False)
    
    
    cpdef getLocalArray(self, Vec gvec, Vec lvec):
        self.globalToLocal(gvec, lvec)
        
        return self.reshape(lvec.getArray(), local=True)
