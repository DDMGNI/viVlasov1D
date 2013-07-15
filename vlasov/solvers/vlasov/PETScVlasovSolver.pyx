'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from vlasov.toolbox.Toolbox import Toolbox


cdef class PETScVlasovSolverBase(object):
    '''
    The PETScSolver class is the base class for all Solver objects
    containing functions to set up the Jacobian matrix, the function
    that constitutes the RHS of the system and possibly a matrix-free
    implementation of the Jacobian.
    '''
    
    def __init__(self, VIDA da1, VIDA dax, Vec H0,
                 npy.ndarray[npy.float64_t, ndim=1] v,
                 npy.uint64_t nx, npy.uint64_t nv,
                 npy.float64_t ht, npy.float64_t hx, npy.float64_t hv,
                 npy.float64_t charge, npy.float64_t coll_freq=0.,
                 regularisation=0.):
        '''
        Constructor
        '''
        
        # distributed array
        self.dax = dax
        self.da1 = da1
        
        # grid
        self.nx = nx
        self.nv = nv
        
        self.ht = ht
        self.hx = hx
        self.hv = hv
        
        self.ht_inv = 1. / self.ht
        self.hx_inv = 1. / self.hx
        self.hv_inv = 1. / self.hv
        
        self.hx2     = hx**2
        self.hx2_inv = 1. / self.hx2 
        
        self.hv2     = hv**2
        self.hv2_inv = 1. / self.hv2 
        
        # Poisson constant
        self.charge = charge
        
        # collision frequency
        self.nu = coll_freq
        
        # regularisation parameter
        self.regularisation = regularisation
        
        # velocity grid
        self.v = v.copy()
        
        # kinetic Hamiltonian
        self.H0 = H0
        
        # create work and history vectors
        self.H1p = self.da1.createGlobalVec()
        self.H1h = self.da1.createGlobalVec()
        self.H2p = self.da1.createGlobalVec()
        self.H2h = self.da1.createGlobalVec()
        
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
        
        # create toolbox object
        self.toolbox = Toolbox(da1, dax, v, nx, nv, ht, hx, hv)
        
        
    
    cpdef update_history(self, Vec F, Vec P, Vec Pext, Vec N, Vec U, Vec E):
        F.copy(self.Fh)
        P.copy(self.Ph)
        N.copy(self.Nh)
        U.copy(self.Uh)
        E.copy(self.Eh)
        
        self.toolbox.compute_collision_factor(self.Nh, self.Uh, self.Eh, self.Ah)
        self.toolbox.potential_to_hamiltonian(self.Ph, self.H1h)
        self.toolbox.potential_to_hamiltonian(Pext,    self.H2h)
        
    
    cpdef update_previous(self, Vec F, Vec P, Vec Pext, Vec N, Vec U, Vec E):
        F.copy(self.Fp)
        P.copy(self.Pp)
        N.copy(self.Np)
        U.copy(self.Up)
        E.copy(self.Ep)
        
        self.toolbox.compute_collision_factor(self.Np, self.Up, self.Ep, self.Ap)
        self.toolbox.potential_to_hamiltonian(self.Pp, self.H1p)
        self.toolbox.potential_to_hamiltonian(Pext,    self.H2p)
        
        
    
    cpdef update_delta(self, Vec F):
        F.copy(self.Fd)
        
        
    
    cpdef snes_mult(self, SNES snes, Vec X, Vec Y):
        self.update_delta(X)
        self.jacobian(Y)
        
    
    cpdef mult(self, Mat mat, Vec X, Vec Y):
        self.update_delta(X)
        self.jacobian(Y)
        
    
    cpdef jacobian_mult(self, Vec X, Vec Y):
        self.update_delta(X)
        self.jacobian(Y)
        
    
    cpdef function_snes_mult(self, SNES snes, Vec X, Vec Y):
        self.update_delta(X)
        self.function(Y)
        
    
    cpdef function_mult(self, Vec X, Vec Y):
        self.update_delta(X)
        self.function(Y)
        
    
    cdef get_data_arrays(self):
        self.h0  = self.da1.getLocalArray(self.H0,  self.localH0 )
        self.h1p = self.da1.getLocalArray(self.H1p, self.localH1p)
        self.h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
        self.h2p = self.da1.getLocalArray(self.H2p, self.localH2p)
        self.h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
        
        self.fp  = self.da1.getLocalArray(self.Fp,  self.localFp)
        self.pp  = self.dax.getLocalArray(self.Pp,  self.localPp)
        self.np  = self.dax.getLocalArray(self.Np,  self.localNp)
        self.up  = self.dax.getLocalArray(self.Up,  self.localUp)
        self.ep  = self.dax.getLocalArray(self.Ep,  self.localEp)
        self.ap  = self.dax.getLocalArray(self.Ap,  self.localAp)
        
        self.fh  = self.da1.getLocalArray(self.Fh,  self.localFh)
        self.ph  = self.dax.getLocalArray(self.Ph,  self.localPh)
        self.nh  = self.dax.getLocalArray(self.Nh,  self.localNh)
        self.uh  = self.dax.getLocalArray(self.Uh,  self.localUh)
        self.eh  = self.dax.getLocalArray(self.Eh,  self.localEh)
        self.ah  = self.dax.getLocalArray(self.Ah,  self.localAh)
        
        self.fd  = self.da1.getLocalArray(self.Fd,  self.localFd)


    cdef get_data_arrays_jacobian(self):
        self.h0  = self.da1.getLocalArray(self.H0,  self.localH0 )
        self.h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
        self.h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
        
#         self.fp  = self.da1.getLocalArray(self.Fp,  self.localFp)
#         self.pp  = self.dax.getLocalArray(self.Pp,  self.localPp)
#         self.np  = self.dax.getLocalArray(self.Np,  self.localNp)
#         self.up  = self.dax.getLocalArray(self.Up,  self.localUp)
#         self.ep  = self.dax.getLocalArray(self.Ep,  self.localEp)
#         self.ap  = self.dax.getLocalArray(self.Ap,  self.localAp)
        
#         self.fh  = self.da1.getLocalArray(self.Fh,  self.localFh)
#         self.ph  = self.dax.getLocalArray(self.Ph,  self.localPh)
#         self.nh  = self.dax.getLocalArray(self.Nh,  self.localNh)
#         self.uh  = self.dax.getLocalArray(self.Uh,  self.localUh)
#         self.eh  = self.dax.getLocalArray(self.Eh,  self.localEh)
#         self.ah  = self.dax.getLocalArray(self.Ah,  self.localAh)
        
        self.fd  = self.da1.getLocalArray(self.Fd,  self.localFd)


    cdef get_data_arrays_function(self):
        self.h0  = self.da1.getLocalArray(self.H0,  self.localH0 )
        self.h1p = self.da1.getLocalArray(self.H1p, self.localH1p)
        self.h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
        self.h2p = self.da1.getLocalArray(self.H2p, self.localH2p)
        self.h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
        
#         self.pp  = self.dax.getLocalArray(self.Pp,  self.localPp)
#         self.np  = self.dax.getLocalArray(self.Np,  self.localNp)
#         self.up  = self.dax.getLocalArray(self.Up,  self.localUp)
#         self.ep  = self.dax.getLocalArray(self.Ep,  self.localEp)
#         self.ap  = self.dax.getLocalArray(self.Ap,  self.localAp)
        
        self.fh  = self.da1.getLocalArray(self.Fh,  self.localFh)
#         self.ph  = self.dax.getLocalArray(self.Ph,  self.localPh)
#         self.nh  = self.dax.getLocalArray(self.Nh,  self.localNh)
#         self.uh  = self.dax.getLocalArray(self.Uh,  self.localUh)
#         self.eh  = self.dax.getLocalArray(self.Eh,  self.localEh)
#         self.ah  = self.dax.getLocalArray(self.Ah,  self.localAh)
        
        self.fd  = self.da1.getLocalArray(self.Fd,  self.localFd)
