'''
Created on Jan 25, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from libc.math cimport exp, pow, sqrt


cdef class Toolbox(object):
    '''
    
    '''
    
    def __cinit__(self,
                 VIDA da1  not None,
                 VIDA dax  not None,
                 Grid grid not None):
        '''
        Constructor
        '''
        
        # distributed arrays and grid
        self.dax  = dax
        self.da1  = da1
        self.grid = grid
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef potential_to_hamiltonian(self, Vec P, Vec H):
        cdef np.uint64_t i
        cdef np.uint64_t pxs, pxe, hxs, hxe, hys, hye
        cdef np.float64_t phisum, phiave
        
        cdef np.ndarray[np.float64_t, ndim=2] h = self.da1.getGlobalArray(H)
        cdef np.ndarray[np.float64_t, ndim=1] p
        cdef np.ndarray[np.float64_t, ndim=1] p0
        
        (pxs, pxe),            = self.dax.getRanges()
        (hxs, hxe), (hys, hye) = self.da1.getRanges()
        
        phisum = P.sum()
        phiave = phisum / self.grid.nx
        p0     = np.empty(hxe-hxs)
        
        if pxs == hxs and pxe == hxe:
            p  = self.dax.getGlobalArray(P)
            p0 = p - phiave
            
            for j in range(0, hye-hys):
                h[:, j] = p0
                
        else:
            scatter, pVec = PETSc.Scatter.toAll(P)
    
            scatter.scatter(P, pVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            
#             p = pVec.getValues(range(hxs, hxe)).copy()
            p  = pVec.getValues(range(0, self.grid.nx)).copy()
            p0 = p[hxs:hxe] - phiave
            
            scatter.destroy()
            pVec.destroy()
            
            for j in range(0, hye-hys):
                h[:, j] = p0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef compute_density(self, Vec F, Vec N):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe, ys, ye 
        cdef np.float64_t n
        
        cdef np.ndarray[np.float64_t, ndim=2] f = self.da1.getGlobalArray(F)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        N.set(0.)
        N.assemble()
        
        for i in range(0, xe-xs):
            n = 0.
             
            for j in range(0, ye-ys):
                n += f[i,j]
                
            N.setValue(i+xs, n*self.grid.hv, addv=PETSc.InsertMode.ADD_VALUES)
    
        N.assemble()
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef compute_velocity_density(self, Vec F, Vec U):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe, ye, ys
        cdef np.float64_t u
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=1] v = self.grid.v
        cdef np.ndarray[np.float64_t, ndim=2] f = self.da1.getGlobalArray(F)
        
        U.set(0.)
        U.assemble()
        
        for i in range(0, xe-xs):
            u = 0.
            
            for j in range(0, ye-ys):
                u += v[j+ys] * f[i,j]
                
            U.setValue(i+xs, u*self.grid.hv, addv=PETSc.InsertMode.ADD_VALUES)
    
        U.assemble()
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef compute_energy_density(self, Vec F, Vec E):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe, ye, ys
        cdef np.float64_t e
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=1] v2 = self.grid.v2
        cdef np.ndarray[np.float64_t, ndim=2] f  = self.da1.getGlobalArray(F)
        
        E.set(0.)
        E.assemble()
        
        for i in range(0, xe-xs):
            e = 0.
            
            for j in range(0, ye-ys):
                e += v2[j+ys] * f[i,j]
            
            E.setValue(i+xs, e*self.grid.hv, addv=PETSc.InsertMode.ADD_VALUES)
    
        E.assemble()
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef compute_collision_factor(self, Vec N, Vec U, Vec E, Vec A):
        cdef np.uint64_t i
        cdef np.uint64_t xs, xe
        
        cdef np.ndarray[np.float64_t, ndim=1] n = self.dax.getGlobalArray(N)
        cdef np.ndarray[np.float64_t, ndim=1] u = self.dax.getGlobalArray(U)
        cdef np.ndarray[np.float64_t, ndim=1] e = self.dax.getGlobalArray(E)
        cdef np.ndarray[np.float64_t, ndim=1] a = self.dax.getGlobalArray(A)
        
        (xs, xe), = self.dax.getRanges()
        
        for i in range(xs, xe):
            a[i] = n[i] * n[i] / (n[i] * e[i] - u[i] * u[i])
            
#             try:
#                 a[i] = n[i] * n[i] / (n[i] * e[i] - u[i] * u[i])
#                 break
#             except ZeroDivisionError:
#                 print("ZeroDivisionError")
#                 print(i, n[i], u[i], e[i], a[i])
            
    
    
    def initialise_kinetic_hamiltonian(self, Vec H, np.float64_t mass):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe, ys, ye
         
        cdef np.ndarray[np.float64_t, ndim=1] v = self.grid.v
        cdef np.ndarray[np.float64_t, ndim=2] h = self.da1.getGlobalArray(H)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for i in range(0, xe-xs):
            for j in range(0, ye-ys):

                h[i, j] = 0.5 * v[j+ys]**2 * mass
 
 
    def initialise_distribution_function(self, Vec F, init_function):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe, ys, ye
        
        cdef np.ndarray[np.float64_t, ndim=2] f = self.da1.getGlobalArray(F)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    f[i-xs, j-ys] = 0.0
                else:
                    f[i-xs, j-ys] = init_function(self.grid.x[i], self.grid.v[j]) 
 
 
    def initialise_distribution_nT(self, Vec F, Vec N, Vec T):
        cdef np.uint64_t i, j
        cdef np.uint64_t xs, xe, ys, ye
         
        cdef np.ndarray[np.float64_t, ndim=1] v = self.grid.v
        cdef np.ndarray[np.float64_t, ndim=2] f = self.da1.getGlobalArray(F)
        cdef np.ndarray[np.float64_t, ndim=1] n = N.getArray()
        cdef np.ndarray[np.float64_t, ndim=1] t = T.getArray()
        
        cdef np.float64_t fac = sqrt(0.5 / np.pi)
         
        (xs, xe), (ys, ye) = self.da1.getRanges()
 
        for i in range(xs, xe):
            for j in range(ys, ye):
                if j < self.grid.stencil or j >= self.grid.nv-self.grid.stencil:
                    f[i-xs, j-ys] = 0.0
                else:
                    f[i-xs, j-ys] = n[i] * fac * exp( - 0.5 * v[j]**2 / T[i] ) / sqrt(T[i])
