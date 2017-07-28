'''
Created on May 27, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from numpy cimport ndarray, float64_t
from numpy  import swapaxes
from petsc4py.PETSc cimport Vec


cdef reshape(object dmda, ndarray vec, bool local):
    if local:
        cstarts, csizes = dmda.getGhostCorners()
    else:
        cstarts, csizes = dmda.getCorners()
    
#         xm, ym, zm = csizes
    
    dim = dmda.getDim()
    dof = dmda.getDof()
     
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


cpdef getGlobalArray(object dmda, Vec tvec):
    return reshape(dmda, tvec.getArray(), local=False)

cpdef getGlobalArrayRO(object dmda, Vec tvec):
    return reshape(dmda, tvec.getArray(readonly=True).copy(), local=False)

cpdef getLocalArray(object dmda, Vec gvec, Vec lvec):
    dmda.globalToLocal(gvec, lvec)
    return reshape(dmda, lvec.getArray(), local=True)

cpdef getLocalArrayRO(object dmda, Vec gvec, Vec lvec):
    dmda.globalToLocal(gvec, lvec)
    return reshape(dmda, lvec.getArray(readonly=True).copy(), local=True)
