'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py.PETSc cimport DA, Mat, Vec

from vlasov.predictor.PETScArakawa import PETScArakawa


cdef class PETScSolver(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DA da1, DA da2, DA dax, DA day, Vec X, Vec B, Vec H0,
                 np.ndarray[np.float64_t, ndim=1] v,
                 np.uint64_t nx, np.uint64_t nv,
                 np.float64_t ht, np.float64_t hx, np.float64_t hv,
                 np.float64_t poisson_const, np.float64_t alpha=0.):
        '''
        Constructor
        '''
        
        assert dax.getDim() == 1
        assert da1.getDim() == 2
        assert da2.getDim() == 2
        
        # distributed array
        self.dax = dax
        self.day = day
        self.da1 = da1
        self.da2 = da2
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv
        
        self.hx2     = hx**2
        self.hx2_inv = 1. / self.hx2 
        
        self.hv2     = hv**2
        self.hv2_inv = 1. / self.hv2 
        
        # velocity grid
        self.v = v.copy()
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # poisson constant
        self.poisson_const = poisson_const
        
        # collision parameter
        self.alpha = alpha
        
        # create work and history vectors
        self.H1  = self.da1.createGlobalVec()
        self.H1h = self.da1.createGlobalVec()
        self.F   = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        self.VF  = self.da1.createGlobalVec()
        self.VFh = self.da1.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1  = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localF   = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        self.localVF  = da1.createLocalVec()
        self.localVFh = da1.createLocalVec()
        
        # create Arakawa solver object
        self.arakawa     = PETScArakawa(da1, nx, nv, hx, hv)
        
    
    def update_history(self, Vec F, Vec H1):
        F.copy(self.Fh)
        H1.copy(self.H1h)
        
        self.calculate_moments(F)
        self.VF.copy(self.VFh)
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
    
    
    @cython.boundscheck(False)
    def calculate_moments(self, Vec F):
        cdef np.uint64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] gf = self.da1.getVecArray(F)[...]
        cdef np.ndarray[np.float64_t, ndim=2] vf = self.da1.getVecArray(self.VF)[...]
        
        for j in np.arange(0, self.nv):
            vf[:, j] = gf[:, j] * self.v[j]
        
        
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        cdef np.float64_t laplace, integral, fsum, phisum
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=2] y  = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=2] x  = self.da2.getVecArray(X)[...]
        cdef np.ndarray[np.float64_t, ndim=2] f  = self.da1.getVecArray(self.F) [...]
        cdef np.ndarray[np.float64_t, ndim=2] h1 = self.da1.getVecArray(self.H1)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] h0
        cdef np.ndarray[np.float64_t, ndim=2] fh
        cdef np.ndarray[np.float64_t, ndim=2] h1h
        cdef np.ndarray[np.float64_t, ndim=1] p
        cdef np.ndarray[np.float64_t, ndim=1] ph
        
        
        # copy x to f
        f[:,:] = x[:, 0:-1]
        
        # copy p to h1
        for j in range(0, self.nv):
            h1[:, j] = x[:, -1]
        
        # calculate average density
        fsum = self.F.sum() * self.hv / self.nx
        
        # calculate moments for collision operator
        self.calculate_moments(self.F)
        
        
        self.da1.globalToLocal(self.H0, self.localH0)
        self.da1.globalToLocal(self.H1, self.localH1)
        self.da1.globalToLocal(self.H1h,self.localH1h)
        self.da1.globalToLocal(self.F,  self.localF)
        self.da1.globalToLocal(self.Fh, self.localFh)
        self.da1.globalToLocal(self.VF, self.localVF)

        h0  = self.da1.getVecArray(self.localH0) [...]
        h1  = self.da1.getVecArray(self.localH1) [...]
        h1h = self.da1.getVecArray(self.localH1h)[...]
        f   = self.da1.getVecArray(self.localF)  [...]
        fh  = self.da1.getVecArray(self.localFh) [...]
        vf  = self.da1.getVecArray(self.localVF) [...]
        p   = h1 [:, 0]
        ph  = h1h[:, 0]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
                
                
                if j == self.nv:
                    # Poisson equation
                    
                    laplace  = (p[ix-1] - 2. * p[ix] + p[ix+1])
                    
                    integral = ( \
                                 + 1. * f[ix-1, :].sum() \
                                 + 2. * f[ix,   :].sum() \
                                 + 1. * f[ix+1, :].sum() \
                               ) * 0.25 * self.hv
                    
                    y[iy, jy] = - laplace + self.poisson_const * (integral-fsum) * self.hx2
                
                
                else:
                    # Vlasov Equation
                    
                    if j == 0 or j == self.nv-1:
                        # Dirichlet Boundary Conditions
                        y[iy, jy] = f[ix, jx]
                        
                    else:
                        y[iy, jy] = self.time_derivative(f, ix, jx) \
                                  + 0.5 * self.arakawa.arakawa(f,  h0,  ix, jx) \
                                  + 0.5 * self.arakawa.arakawa(f,  h1h, ix, jx) \
                                  + 0.5 * self.arakawa.arakawa(fh, h1,  ix, jx) \
                                  - 0.5 * self.alpha * self.dvdv(f,  ix, jx) \
                                  - 0.5 * self.alpha * self.coll(vf, ix, jx)
                    
        
    
    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        self.da1.globalToLocal(self.H0,  self.localH0)
        self.da1.globalToLocal(self.Fh,  self.localFh)
        self.da1.globalToLocal(self.VFh, self.localVFh)
        
        cdef np.ndarray[np.float64_t, ndim=2] b   = self.da2.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0  = self.da1.getVecArray(self.localH0 )[...]
        cdef np.ndarray[np.float64_t, ndim=2] fh  = self.da1.getVecArray(self.localFh )[...]
        cdef np.ndarray[np.float64_t, ndim=2] vfh = self.da1.getVecArray(self.localVFh)[...]
        
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
                
                if j == self.nv:
                    # Poisson equation
                    
                    b[iy, jy] = 0.
            
                
                else:
                    # Vlasov equation
                
                    if j == 0 or j == self.nv-1:
                        # Dirichlet boundary conditions
                        b[iy, jy] = 0.0
                        
                    else:
                        b[iy, jy] = self.time_derivative(fh, ix, jx) \
                                  - 0.5 * self.arakawa.arakawa(fh, h0, ix, jx) \
                                  + 0.5 * self.alpha * self.dvdv(fh,  ix, jx) \
                                  + 0.5 * self.alpha * self.coll(vfh, ix, jx)
    


    @cython.boundscheck(False)
    cdef np.float64_t time_derivative(self, np.ndarray[np.float64_t, ndim=2] x,
                                            np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * x[i-1, j-1] \
                   + 2. * x[i-1, j  ] \
                   + 1. * x[i-1, j+1] \
                   + 2. * x[i,   j-1] \
                   + 4. * x[i,   j  ] \
                   + 2. * x[i,   j+1] \
                   + 1. * x[i+1, j-1] \
                   + 2. * x[i+1, j  ] \
                   + 1. * x[i+1, j+1] \
                 ) / (16. * self.ht)
        
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t dvdv(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     + 1. * x[i-1, j-1] \
                     + 1. * x[i-1, j+1] \
                     - 2. * x[i-1, j  ] \
                     + 1. * x[i+1, j-1] \
                     + 1. * x[i+1, j+1] \
                     - 2. * x[i+1, j  ] \
                     + 2. * x[i,   j-1] \
                     + 2. * x[i,   j+1] \
                     - 4. * x[i,   j  ] \
                 ) * 0.25 * self.hv2_inv
        
        return result
    
    
    
    @cython.boundscheck(False)
    cdef np.float64_t coll(self, np.ndarray[np.float64_t, ndim=2] x,
                                 np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                   + 1. * ( x[i-1, j+1] - x[i-1, j-1] ) \
                   + 2. * ( x[i,   j+1] - x[i,   j-1] ) \
                   + 1. * ( x[i+1, j+1] - x[i+1, j-1] ) \
                 ) * 0.25 / (2. * self.hv)
        
        return result
    
    
