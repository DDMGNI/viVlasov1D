'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

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
                 double charge=-1.,
                 double coll_freq=0.,
                 double coll_diff=1.,
                 double coll_drag=1.,
                 double regularisation=0.):
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
        self.localFp  = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        self.localFd  = da1.createLocalVec()
        
        self.localFave = self.da1.createLocalVec()
        self.localHave = self.da1.createLocalVec()
        
        
    def __dealloc__(self):
        self.localFp.destroy()
        self.localFh.destroy()
        self.localFd.destroy()
        
        self.localFave.destroy()
        self.localHave.destroy()
        
        self.Fp.destroy()
        self.Fh.destroy()
    
        self.Fave.destroy()
        self.Have.destroy()
        
    
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
        
