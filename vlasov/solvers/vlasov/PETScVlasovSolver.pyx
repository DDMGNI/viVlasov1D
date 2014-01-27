'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc


cdef class PETScVlasovSolverBase(object):
    '''
    The PETScSolver class is the base class for all Solver objects
    containing functions to set up the Jacobian matrix, the function
    that constitutes the RHS of the system and possibly a matrix-free
    implementation of the Jacobian.
    '''
    
    def __init__(self,
                 VIDA da1  not None,
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
                 npy.float64_t regularisation=0.):
        '''
        Constructor
        '''
        
        # distributed arrays and grid
        self.da1  = da1
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
        
        # distribution function
        self.Fp  = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        
        # averages
        self.Fave = self.da1.createGlobalVec()
        self.Have = self.da1.createGlobalVec()
        
        # moments
        self.Np  = None
        self.Up  = None
        self.Ep  = None
        self.Ap  = None
        
        self.Nh  = None
        self.Uh  = None
        self.Eh  = None
        self.Ah  = None
        
        # create local vectors
#         self.localH0  = da1.createLocalVec()
#         self.localH1p = da1.createLocalVec()
#         self.localH1h = da1.createLocalVec()
#         self.localH2p = da1.createLocalVec()
#         self.localH2h = da1.createLocalVec()

        self.localFp  = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        self.localFd  = da1.createLocalVec()
        
        self.localFave = self.da1.createLocalVec()
        self.localHave = self.da1.createLocalVec()
        
        
    def __dealloc__(self):
        self.Fp.destroy()
        self.Fh.destroy()
    
        self.Fave.destroy()
        self.Have.destroy()
        
        self.localFp.destroy()
        self.localFh.destroy()
        self.localFd.destroy()
        
        self.localFave.destroy()
        self.localHave.destroy()
        
    
    def set_moments(self, Vec Np, Vec Up, Vec Ep, Vec Ap, Vec Nh, Vec Uh, Vec Eh, Vec Ah):
        self.Np = Np
        self.Up = Up
        self.Ep = Ep
        self.Ap = Ap
        
        self.Nh = Nh
        self.Uh = Uh
        self.Eh = Eh
        self.Ah = Ah
        
    
    def update_history(self, Vec F):
        F.copy(self.Fh)
    
    def update_previous(self, Vec F):
        F.copy(self.Fp)
        
        self.H0.copy(self.Have)
        self.Have.axpy(.5, self.H1p)
        self.Have.axpy(.5, self.H1h)
        self.Have.axpy(.5, self.H2p)
        self.Have.axpy(.5, self.H2h)
        
    
    cpdef snes_mult(self, SNES snes, Vec X, Vec Y):
        self.jacobian(X, Y)
        
    
    cpdef mult(self, Mat mat, Vec X, Vec Y):
        self.jacobian(X, Y)
        
    
    cpdef jacobian_mult(self, Vec X, Vec Y):
        self.jacobian(X, Y)
        
    
    cpdef function_snes_mult(self, SNES snes, Vec X, Vec Y):
        self.function(X, Y)
        
    
    cpdef function_mult(self, Vec X, Vec Y):
        self.function(X, Y)
        
    
#     cdef get_data_arrays(self):
#         self.h0  = self.da1.getLocalArray(self.H0,  self.localH0 )
#         self.h1p = self.da1.getLocalArray(self.H1p, self.localH1p)
#         self.h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
#         self.h2p = self.da1.getLocalArray(self.H2p, self.localH2p)
#         self.h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
#         
#         self.fp  = self.da1.getLocalArray(self.Fp,  self.localFp)
#         self.fh  = self.da1.getLocalArray(self.Fh,  self.localFh)
#         
#         self.np  = self.Np.getArray()
#         self.up  = self.Up.getArray()
#         self.ep  = self.Ep.getArray()
#         self.ap  = self.Ap.getArray()
#         
#         self.nh  = self.Nh.getArray()
#         self.uh  = self.Uh.getArray()
#         self.eh  = self.Eh.getArray()
#         self.ah  = self.Ah.getArray()
# 
# 
#     cdef get_data_arrays_jacobian(self):
#         self.h0  = self.da1.getLocalArray(self.H0,  self.localH0 )
#         self.h1p = self.da1.getLocalArray(self.H1p, self.localH1p)
#         self.h1h = self.da1.getLocalArray(self.H1h, self.localH1h)
#         self.h2p = self.da1.getLocalArray(self.H2p, self.localH2p)
#         self.h2h = self.da1.getLocalArray(self.H2h, self.localH2h)
