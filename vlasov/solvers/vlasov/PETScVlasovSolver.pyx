'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

from petsc4py import PETSc

from vlasov.solvers.components.CollisionOperator import CollisionOperator
from vlasov.solvers.components.DoubleBracket     import DoubleBracket
from vlasov.solvers.components.PoissonBracket    import PoissonBracket
from vlasov.solvers.components.Regularisation    import Regularisation
from vlasov.solvers.components.TimeDerivative    import TimeDerivative
from vlasov.solvers.preconditioner.TensorProduct import TensorProductPreconditioner


cdef class PETScVlasovSolverBase(object):
    '''
    The PETScSolver class is the base class for all Solver objects
    containing functions to set up the Jacobian matrix, the function
    that constitutes the RHS of the system and possibly a matrix-free
    implementation of the Jacobian.
    '''
    
    def __init__(self,
                 config    not None,
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
        
        # distribution function and Hamiltonian
        self.Fp  = self.da1.createGlobalVec()
        self.Fh  = self.da1.createGlobalVec()
        self.Hp  = self.da1.createGlobalVec()
        self.Hh  = self.da1.createGlobalVec()
        
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
        
        # interim vector
        self.X   = self.da1.createGlobalVec()
        
        # create local vectors
        self.localFp  = da1.createLocalVec()
        self.localFh  = da1.createLocalVec()
        self.localFd  = da1.createLocalVec()
        
        self.localFave = self.da1.createLocalVec()
        self.localFder = self.da1.createLocalVec()
        self.localHave = self.da1.createLocalVec()
        
        # create components
        self.time_derivative    = TimeDerivative.create(config.get_averaging_operator(), da1, grid)
        self.poisson_bracket    = PoissonBracket.create(config.get_poisson_bracket(), da1, grid)
        self.double_bracket     = DoubleBracket.create(config.get_double_bracket(), da1, grid, self.poisson_bracket, coll_freq)
        self.collision_operator = CollisionOperator.create(config.get_collision_operator(), da1, grid, coll_freq, coll_diff, coll_drag)
        self.regularisation     = Regularisation(config, da1, grid, regularisation)
        self.preconditioner     = TensorProductPreconditioner.create(config.get_preconditioner(), da1, grid)
        
        

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
        
        self.H0.copy(self.Hp)
        self.Hp.axpy(.5, self.H1p)
        self.Hp.axpy(.5, self.H2p)

        self.H0.copy(self.Hh)
        self.Hh.axpy(.5, self.H1h)
        self.Hh.axpy(.5, self.H2h)
        
    
    cpdef jacobian(self, Vec F, Vec Y):
        if self.preconditioner == None:
            self.jacobian_solver(F, Y)
        else:
            self.jacobian_solver(F, self.X)
            self.preconditioner.tensorProduct(self.X, Y)
    
    
    cpdef function(self, Vec F, Vec Y):
        if self.preconditioner == None:
            self.function_solver(F, Y)
        else:
            self.function_solver(F, self.X)
            self.preconditioner.tensorProduct(self.X, Y)
        
    
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
    
    
    cpdef double function_norm(self, Vec F, Vec Y):
        self.function_solver(F, Y)
        return Y.norm()
        

    cdef jacobian_solver(self, Vec F, Vec Y):
        print("ERROR: function not implemented.")
 
    cdef function_solver(self, Vec F, Vec Y):
        print("ERROR: function not implemented.")
