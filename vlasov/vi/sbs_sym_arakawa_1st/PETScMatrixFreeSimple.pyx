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
        
        # save solution and RHS vector
        self.X = X
        self.B = B
        
        # create history vectors
        self.X1  = self.da2.createGlobalVec()
        self.F   = self.da1.createGlobalVec()
        self.PHI = self.dax.createGlobalVec()

        self.VF  = self.da1.createGlobalVec()
        self.VF1 = self.da1.createGlobalVec()
        
#        self.N   = self.dax.createGlobalVec()
#        self.U   = self.dax.createGlobalVec()
#        self.E   = self.dax.createGlobalVec()
#        
#        self.A1  = self.dax.createGlobalVec()
#        self.A1h = self.dax.createGlobalVec()
#        self.A2  = self.dax.createGlobalVec()
#        self.A2h = self.dax.createGlobalVec()
        
        # create local vectors
        self.localB   = da2.createLocalVec()
        self.localX   = da2.createLocalVec()
        self.localX1  = da2.createLocalVec()
        self.localH0  = da1.createLocalVec()

        self.localVF  = da1.createLocalVec()
        self.localVF1 = da1.createLocalVec()

#        self.localA1  = dax.createLocalVec()
#        self.localA1h = dax.createLocalVec()
#        self.localA2  = dax.createLocalVec()
#        self.localA2h = dax.createLocalVec()
        
        # create Arakawa solver object
        self.arakawa     = PETScArakawa(da1, da2, nx, nv, hx, hv)
        
    
    def update_history(self, Vec X):
        X.copy(self.X1)
        self.VF.copy(self.VF1)
        
#        self.A1.copy(self.A1h)
#        self.A2.copy(self.A2h)
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    
#    @cython.boundscheck(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef np.uint64_t i, j
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xe, xs, ye, ys
        
        cdef np.float64_t laplace, integral, fsum, phisum
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        self.da2.globalToLocal(X,       self.localX)
        self.da2.globalToLocal(self.X1, self.localX1)
        self.da1.globalToLocal(self.H0, self.localH0)

        cdef np.ndarray[np.float64_t, ndim=3] y  = self.da2.getVecArray(Y)[...]
        cdef np.ndarray[np.float64_t, ndim=3] x  = self.da2.getVecArray(self.localX) [...]
        cdef np.ndarray[np.float64_t, ndim=3] xh = self.da2.getVecArray(self.localX1)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] f  = x [:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] fh = xh[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] p  = x [:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] ph = xh[:,:,1]
        
        cdef np.ndarray[np.float64_t, ndim=2] gf = self.da2.getVecArray(self.X)[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] vf = self.da1.getVecArray(self.VF)[...]
        
        for j in np.arange(0, ye-ys):
            vf[:, j] = gf[:, j] * self.v[j]
        
#        cdef np.ndarray[np.float64_t, ndim=2] tu = np.empty_like(gf)
#        cdef np.ndarray[np.float64_t, ndim=2] te = np.empty_like(gf)
#        
#        cdef np.ndarray[np.float64_t, ndim=1] n = self.dax.getVecArray(self.N)[...]
#        cdef np.ndarray[np.float64_t, ndim=1] u = self.dax.getVecArray(self.U)[...]
#        cdef np.ndarray[np.float64_t, ndim=1] e = self.dax.getVecArray(self.E)[...]
#        
#        for j in np.arange(0, ye-ys):
#            tu[:, j] = gf[:, j] * self.v[j]
#            te[:, j] = tu[:, j] * self.v[j]
#        
#        n[:] = gf.sum(axis=1) * self.hv
#        u[:] = tu.sum(axis=1) * self.hv
#        e[:] = te.sum(axis=1) * self.hv
#        
#        cdef np.ndarray[np.float64_t, ndim=1] a1  = self.dax.getVecArray(self.A1)[...]
#        cdef np.ndarray[np.float64_t, ndim=1] a2  = self.dax.getVecArray(self.A2)[...]
#        
#        a1[:] = u
#        a2[:] = e
#        
#        self.dax.globalToLocal(self.A1,  self.localA1 )
#        self.dax.globalToLocal(self.A1h, self.localA1h)
#        self.dax.globalToLocal(self.A2,  self.localA2 )
#        self.dax.globalToLocal(self.A2h, self.localA2h)
#        
#        a1 = self.dax.getVecArray(self.localA1)[...]
#        a2 = self.dax.getVecArray(self.localA2)[...]
#        
#        cdef np.ndarray[np.float64_t, ndim=1] a1h = self.dax.getVecArray(self.localA1h)[...]
#        cdef np.ndarray[np.float64_t, ndim=1] a2h = self.dax.getVecArray(self.localA2h)[...]
        
        
        self.da1.globalToLocal(self.VF, self.localVF)
        vf = self.da1.getVecArray(self.localVF)[...]
        
        
        gx  = self.da2.getVecArray(X)
        
        phi = self.dax.getVecArray(self.PHI)
        phi[xs:xe] = gx[:, 0, 1]
        phisum = self.PHI.sum()
        
        tf = self.da1.getVecArray(self.F)
        tf[xs:xe, ys:ye] = gx[:, :, 0]
        fsum = self.F.sum() * self.hv / self.nx
        
#        print(fsum)
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
#            if i == 0:
#                y[iy, :, 1] = phisum
#                
#            else:
            laplace  = (p[ix-1, 0] - 2. * p[ix, 0] + p[ix+1, 0])
            
            integral = ( \
                         + 1. * f[ix-1, :].sum() \
                         + 2. * f[ix,   :].sum() \
                         + 1. * f[ix+1, :].sum() \
                       ) * 0.25 * self.hv
            
            y[iy, :, 1] = - laplace + self.poisson_const * (integral-fsum) * self.hx2
            
            # Vlasov Equation
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
            
                if j == 0 or j == self.nv-1:
                    # Dirichlet Boundary Conditions
                    y[iy, jy, 0] = f[ix, jx]
                    
                else:
                    y[iy, jy, 0] = self.time_derivative(f, ix, jx) \
                                 + 0.5 * self.arakawa.arakawa(f,  h0, ix, jx) \
                                 + 0.5 * self.arakawa.arakawa(f,  ph, ix, jx) \
                                 + 0.5 * self.arakawa.arakawa(fh, p,  ix, jx) \
                                 - 0.5 * self.alpha * self.dvdv(f,  ix, jx) \
                                 - 0.5 * self.alpha * self.C1(vf, ix, jx)
#                                 + 0.5 * self.alpha * self.C1(f,  a1h, ix, jx) \
#                                 + 0.5 * self.alpha * self.C1(fh, a1,  ix, jx) #\
#                                 + 0.5 * self.alpha * self.C2(f,  a2h, self.v, ix, jx) \
#                                 + 0.5 * self.alpha * self.C2(fh, a2,  self.v, ix, jx)
                    
        
    
#    @cython.boundscheck(False)
    def formRHS(self, Vec B):
        cdef np.uint64_t ix, iy, jx, jy
        cdef np.uint64_t xs, xe, ys, ye
        
        self.da2.globalToLocal(self.X1, self.localX1)
        self.da1.globalToLocal(self.H0, self.localH0)
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        cdef np.ndarray[np.float64_t, ndim=3] b  = self.da2.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=3] xh = self.da2.getVecArray(self.localX1)[...]
        cdef np.ndarray[np.float64_t, ndim=2] h0 = self.da1.getVecArray(self.localH0)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] fh = xh[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] ph = xh[:,:,1]
        
        self.da1.globalToLocal(self.VF1, self.localVF1)
        cdef np.ndarray[np.float64_t, ndim=2] vfh = self.da1.getVecArray(self.localVF1)[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            iy = i-xs
            
            # Poisson equation
            b[iy, :, 1] = 0.
            
            
            # Vlasov equation
            for j in np.arange(ys, ye):
                jx = j-ys
                jy = j-ys
                
                if j == 0 or j == self.nv-1:
                    # Dirichlet boundary conditions
                    b[iy, jy, 0] = 0.0
                    
                else:
                    b[iy, jy, 0] = self.time_derivative(fh, ix, jx) \
                                 - 0.5 * self.arakawa.arakawa(fh, h0, ix, jx) \
                                 + 0.5 * self.alpha * self.dvdv(fh, ix, jx) \
                                 + 0.5 * self.alpha * self.C1(vfh, ix, jx)
    


#    @cython.boundscheck(False)
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
    
    
    
#    @cython.boundscheck(False)
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
    
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t C1(self, np.ndarray[np.float64_t, ndim=2] x,
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
    
    
#    @cython.boundscheck(False)
    cdef np.float64_t C2(self, np.ndarray[np.float64_t, ndim=2] x,
                               np.ndarray[np.float64_t, ndim=1] A,
                               np.ndarray[np.float64_t, ndim=1] V,
                               np.uint64_t i, np.uint64_t j):
        '''
        Time Derivative
        '''
        
        cdef np.float64_t result
        
        result = ( \
                     - 1 * x[i-1, j-1] * A[i-1] * V[j-1] \
                     - 1 * x[i-1, j-1] * A[i-1] * V[j  ] \
                     - 1 * x[i-1, j-1] * A[i  ] * V[j-1] \
                     - 1 * x[i-1, j-1] * A[i  ] * V[j  ] \
                     + 1 * x[i-1, j+1] * A[i-1] * V[j+1] \
                     + 1 * x[i-1, j+1] * A[i-1] * V[j  ] \
                     + 1 * x[i-1, j+1] * A[i  ] * V[j+1] \
                     + 1 * x[i-1, j+1] * A[i  ] * V[j  ] \
                     - 1 * x[i+1, j-1] * A[i+1] * V[j-1] \
                     - 1 * x[i+1, j-1] * A[i+1] * V[j  ] \
                     - 1 * x[i+1, j-1] * A[i  ] * V[j-1] \
                     - 1 * x[i+1, j-1] * A[i  ] * V[j  ] \
                     + 1 * x[i+1, j+1] * A[i+1] * V[j+1] \
                     + 1 * x[i+1, j+1] * A[i+1] * V[j  ] \
                     + 1 * x[i+1, j+1] * A[i  ] * V[j+1] \
                     + 1 * x[i+1, j+1] * A[i  ] * V[j  ] \
                     + 1 * x[i-1, j  ] * A[i-1] * V[j+1] \
                     + 1 * x[i-1, j  ] * A[i  ] * V[j+1] \
                     - 1 * x[i-1, j  ] * A[i-1] * V[j-1] \
                     - 1 * x[i-1, j  ] * A[i  ] * V[j-1] \
                     + 1 * x[i+1, j  ] * A[i+1] * V[j+1] \
                     + 1 * x[i+1, j  ] * A[i  ] * V[j+1] \
                     - 1 * x[i+1, j  ] * A[i+1] * V[j-1] \
                     - 1 * x[i+1, j  ] * A[i  ] * V[j-1] \
                     - 1 * x[i,   j-1] * A[i-1] * V[j-1] \
                     - 1 * x[i,   j-1] * A[i-1] * V[j  ] \
                     - 1 * x[i,   j-1] * A[i+1] * V[j-1] \
                     - 1 * x[i,   j-1] * A[i+1] * V[j  ] \
                     - 2 * x[i,   j-1] * A[i  ] * V[j-1] \
                     - 2 * x[i,   j-1] * A[i  ] * V[j  ] \
                     + 1 * x[i,   j+1] * A[i-1] * V[j+1] \
                     + 1 * x[i,   j+1] * A[i-1] * V[j  ] \
                     + 1 * x[i,   j+1] * A[i+1] * V[j+1] \
                     + 1 * x[i,   j+1] * A[i+1] * V[j  ] \
                     + 2 * x[i,   j+1] * A[i  ] * V[j+1] \
                     + 2 * x[i,   j+1] * A[i  ] * V[j  ] \
                     - 1 * x[i,   j  ] * A[i-1] * V[j-1] \
                     + 1 * x[i,   j  ] * A[i-1] * V[j+1] \
                     - 1 * x[i,   j  ] * A[i+1] * V[j-1] \
                     + 1 * x[i,   j  ] * A[i+1] * V[j+1] \
                     - 2 * x[i,   j  ] * A[i  ] * V[j-1] \
                     + 2 * x[i,   j  ] * A[i  ] * V[j+1] \
                   ) / (32. * self.hv)
        
        return result
    
    
    
    
    
    def isSparse(self):
        return False
    


