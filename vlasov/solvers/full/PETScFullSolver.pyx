'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from vlasov.toolbox.Toolbox import Toolbox


cdef class PETScFullSolverBase(object):
    '''
    The PETScSolver class is the base class for all Solver objects
    containing functions to set up the Jacobian matrix, the function
    that constitutes the RHS of the system and possibly a matrix-free
    implementation of the Jacobian.  
    '''
    
    def __init__(self,
                 object da1  not None,
                 object da2  not None,
                 object dax  not None,
                 Grid grid not None,
                 Vec H0  not None,
                 Vec H1p not None,
                 Vec H1h not None,
                 Vec H2p not None,
                 Vec H2h not None,
                 npy.float64_t charge=-1.,
                 npy.float64_t coll_freq=0.,
                 npy.float64_t coll_diff=1.,
                 npy.float64_t coll_drag=1.,
                 regularisation=0.):
        '''
        Constructor
        '''
        
        # distributed array
        self.dax = dax
        self.da1 = da1
        self.da2 = da2
        self.grid = grid
        
        
        # charge
        self.charge = charge
        
        # collision operator
        self.nu = coll_freq
        
        self.coll_diff = coll_diff
        self.coll_drag = coll_drag
        
        # regularisation parameter
        self.regularisation = regularisation

        
        # Hamiltonians
        self.H0  = H0
        self.H1p = H1p
        self.H1h = H1h
        self.H2p = H2p
        self.H2h = H2h
        
        
        # create work and history vectors
        self.H1d = self.da1.createGlobalVec()
        
        self.Fp  = self.da1.createGlobalVec()
        self.Pp  = self.dax.createGlobalVec()
        self.Np  = self.dax.createGlobalVec()
        self.Up  = self.dax.createGlobalVec()
        self.Ep  = self.dax.createGlobalVec()
        self.Ap  = self.dax.createGlobalVec()
        
        self.Fh  = self.da1.createGlobalVec()
        self.Ph  = self.dax.createGlobalVec()
        self.Nh  = self.dax.createGlobalVec()
        self.Uh  = self.dax.createGlobalVec()
        self.Eh  = self.dax.createGlobalVec()
        self.Ah  = self.dax.createGlobalVec()
        
        self.Fd  = self.da1.createGlobalVec()
        self.Pd  = self.dax.createGlobalVec()
        self.Nd  = self.dax.createGlobalVec()
        self.Ud  = self.dax.createGlobalVec()
        self.Ed  = self.dax.createGlobalVec()
        self.Ad  = self.dax.createGlobalVec()
        
        self.Nc  = self.dax.createGlobalVec()
        self.Uc  = self.dax.createGlobalVec()
        self.Ec  = self.dax.createGlobalVec()
        
        # create local vectors
        self.localH0  = da1.createLocalVec()
        self.localH1p = da1.createLocalVec()
        self.localH1h = da1.createLocalVec()
        self.localH1d = da1.createLocalVec()
        self.localH2p = da1.createLocalVec()
        self.localH2h = da1.createLocalVec()

        self.localFp  = da1.createLocalVec()
        self.localPp  = dax.createLocalVec()
        self.localNp  = dax.createLocalVec()
        self.localUp  = dax.createLocalVec()
        self.localEp  = dax.createLocalVec()
        self.localAp  = dax.createLocalVec()
        
        self.localFh  = da1.createLocalVec()
        self.localPh  = dax.createLocalVec()
        self.localNh  = dax.createLocalVec()
        self.localUh  = dax.createLocalVec()
        self.localEh  = dax.createLocalVec()
        self.localAh  = dax.createLocalVec()
        
        self.localFd  = da1.createLocalVec()
        self.localPd  = dax.createLocalVec()
        self.localNd  = dax.createLocalVec()
        self.localUd  = dax.createLocalVec()
        self.localEd  = dax.createLocalVec()
        self.localAd  = dax.createLocalVec()
        
        self.localNc  = dax.createLocalVec()
        self.localUc  = dax.createLocalVec()
        self.localEc  = dax.createLocalVec()
        
        # create toolbox object
        self.toolbox = Toolbox(da1, dax, grid)
        
        
    
    def update_history(self, Vec X):
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        x = self.da2.getVecArray(X)
        f = self.da1.getVecArray(self.Fh)
        p = self.dax.getVecArray(self.Ph)
        n = self.dax.getVecArray(self.Nh)
        u = self.dax.getVecArray(self.Uh)
        e = self.dax.getVecArray(self.Eh)
        
        f[xs:xe] = x[xs:xe, 0:self.grid.nv]
        p[xs:xe] = x[xs:xe,   self.grid.nv]
        n[xs:xe] = x[xs:xe,   self.grid.nv+1]
        u[xs:xe] = x[xs:xe,   self.grid.nv+2]
        e[xs:xe] = x[xs:xe,   self.grid.nv+3]
        
        self.toolbox.compute_collision_factor(self.Nh, self.Uh, self.Eh, self.Ah)
        self.toolbox.potential_to_hamiltonian(self.Ph, self.H1h)
        
    
    def update_previous(self, Vec X):
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        x = self.da2.getVecArray(X)
        f = self.da1.getVecArray(self.Fp)
        p = self.dax.getVecArray(self.Pp)
        n = self.dax.getVecArray(self.Np)
        u = self.dax.getVecArray(self.Up)
        e = self.dax.getVecArray(self.Ep)
        
        f[xs:xe] = x[xs:xe, 0:self.grid.nv]
        p[xs:xe] = x[xs:xe,   self.grid.nv]
        n[xs:xe] = x[xs:xe,   self.grid.nv+1]
        u[xs:xe] = x[xs:xe,   self.grid.nv+2]
        e[xs:xe] = x[xs:xe,   self.grid.nv+3]
        
        self.toolbox.compute_collision_factor(self.Np, self.Up, self.Ep, self.Ap)
        self.toolbox.potential_to_hamiltonian(self.Pp, self.H1p)
        
        
    
    def update_delta(self, Vec X):
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        x = self.da2.getVecArray(X)
        f = self.da1.getVecArray(self.Fd)
        p = self.dax.getVecArray(self.Pd)
        n = self.dax.getVecArray(self.Nd)
        u = self.dax.getVecArray(self.Ud)
        e = self.dax.getVecArray(self.Ed)
        
        f[xs:xe] = x[xs:xe, 0:self.grid.nv]
        p[xs:xe] = x[xs:xe,   self.grid.nv]
        n[xs:xe] = x[xs:xe,   self.grid.nv+1]
        u[xs:xe] = x[xs:xe,   self.grid.nv+2]
        e[xs:xe] = x[xs:xe,   self.grid.nv+3]
        
        self.toolbox.compute_collision_factor(self.Nd, self.Ud, self.Ed, self.Ad)
        self.toolbox.potential_to_hamiltonian(self.Pd, self.H1d)
        
        
    
    def update_external(self, Vec Pext):
        self.H2p.copy(self.H2h)
        self.toolbox.potential_to_hamiltonian(Pext, self.H2p)
        
    
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.update_delta(X)
        self.jacobian(Y)
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.update_delta(X)
        self.jacobian(Y)
        
    
    def function_snes_mult(self, SNES snes, Vec X, Vec Y):
        self.update_delta(X)
        self.function(Y)
        
    
    def function_mult(self, Vec X, Vec Y):
        self.update_delta(X)
        self.function(Y)
        
    
    cdef get_data_arrays(self):
        self.h0  = getLocalArray(self.da1, self.H0,  self.localH0 )
        self.h1p = getLocalArray(self.da1, self.H1p, self.localH1p)
        self.h1h = getLocalArray(self.da1, self.H1h, self.localH1h)
        self.h1d = getLocalArray(self.da1, self.H1d, self.localH1d)
        self.h2p = getLocalArray(self.da1, self.H2p, self.localH2p)
        self.h2h = getLocalArray(self.da1, self.H2h, self.localH2h)
        
        self.fp  = getLocalArray(self.da1, self.Fp,  self.localFp)
        self.pp  = getLocalArray(self.dax, self.Pp,  self.localPp)
        self.np  = getLocalArray(self.dax, self.Np,  self.localNp)
        self.up  = getLocalArray(self.dax, self.Up,  self.localUp)
        self.ep  = getLocalArray(self.dax, self.Ep,  self.localEp)
        self.ap  = getLocalArray(self.dax, self.Ap,  self.localAp)
        
        self.fh  = getLocalArray(self.da1, self.Fh,  self.localFh)
        self.ph  = getLocalArray(self.dax, self.Ph,  self.localPh)
        self.nh  = getLocalArray(self.dax, self.Nh,  self.localNh)
        self.uh  = getLocalArray(self.dax, self.Uh,  self.localUh)
        self.eh  = getLocalArray(self.dax, self.Eh,  self.localEh)
        self.ah  = getLocalArray(self.dax, self.Ah,  self.localAh)
        
        self.fd  = getLocalArray(self.da1, self.Fd,  self.localFd)
        self.pd  = getLocalArray(self.dax, self.Pd,  self.localPd)
        self.nd  = getLocalArray(self.dax, self.Nd,  self.localNd)
        self.ud  = getLocalArray(self.dax, self.Ud,  self.localUd)
        self.ed  = getLocalArray(self.dax, self.Ed,  self.localEd)
        self.ad  = getLocalArray(self.dax, self.Ad,  self.localAd)
        
        self.nc  = getLocalArray(self.dax, self.Nc,  self.localNc)
        self.uc  = getLocalArray(self.dax, self.Uc,  self.localUc)
        self.ec  = getLocalArray(self.dax, self.Ec,  self.localEc)
