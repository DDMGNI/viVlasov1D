'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

from petsc4py import PETSc

from vlasov.solvers.components.Collisions     import Collisions
from vlasov.solvers.components.PoissonBracket import PoissonBracket
from vlasov.solvers.components.Regularisation import Regularisation
from vlasov.solvers.components.TimeDerivative import TimeDerivative


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
        self.Fder = self.da1.createGlobalVec()
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
        self.localFder = self.da1.createLocalVec()
        self.localHave = self.da1.createLocalVec()
        
        # create components
        self.poisson_bracket = PoissonBracket(self.da1, self.grid)
        self.time_derivative = TimeDerivative(self.da1, self.grid)
        self.collisions      = Collisions(self.da1, self.grid, coll_freq, coll_diff, coll_drag)
        self.regularisation  = Regularisation(self.da1, self.grid, regularisation)
        
        
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
        
    
    cpdef jacobian(self, Vec F, Vec Y):
        self.jacobian_solver(F, Y)
    
    
    cpdef function(self, Vec F, Vec Y):
        self.function_solver(F, Y)
        
    
    cpdef snes_mult(self, SNES snes, Vec F, Vec Y):
        self.jacobian(F, Y)
        
    
    cpdef mult(self, Mat mat, Vec F, Vec Y):
        self.jacobian(F, Y)
        
    
    cpdef jacobian_mult(self, Vec F, Vec Y):
        self.jacobian(F, Y)
        
    
    cpdef function_snes_mult(self, SNES snes, Vec F, Vec Y):
        self.function(F, Y)
        
    
    cpdef function_mult(self, Vec F, Vec Y):
        self.function(F, Y)
        

    cdef jacobian_solver(self, Vec F, Vec Y):
        Y.set(0.)
        
        self.call_poisson_bracket(F, self.Have, Y, 0.5)
        self.call_time_derivative(F, Y)
        self.call_collision_operator(F, Y, self.Np, self.Up, self.Ep, self.Ap, 0.5)
        self.call_regularisation(F, Y, 1.0)
    
    
    cdef function_solver(self, Vec F, Vec Y):
        self.Fave.set(0.)
        self.Fave.axpy(0.5, self.Fh)
        self.Fave.axpy(0.5, F)
        
        self.Fder.set(0.)
        self.Fder.axpy(+1, F)
        self.Fder.axpy(-1, self.Fh)
        
        Y.set(0.)
        
        self.call_poisson_bracket(self.Fave, self.Have, Y, 1.0)
        self.call_time_derivative(self.Fder, Y)
        self.call_collision_operator(F, Y, self.Np, self.Up, self.Ep, self.Ap, 0.5)
        self.call_collision_operator(F, Y, self.Nh, self.Uh, self.Eh, self.Ah, 0.5)
        
    
    cdef call_poisson_bracket(self, Vec F, Vec H, Vec Y, double factor):
        print("ERROR: function not implemented.")
    
    cdef call_time_derivative(self, Vec F, Vec Y):
        self.time_derivative.time_derivative(F, Y)
        
    cdef call_collision_operator(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor):
        self.collisions.collT(F, Y, N, U, E, A, factor)
    
    cdef call_regularisation(self, Vec F, Vec Y, double factor):
        self.regularisation.regularisation(F, Y, factor)

