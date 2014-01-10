'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys
import petsc4py
petsc4py.init(sys.argv)

import numpy as np

from petsc4py import PETSc

from vlasov.toolbox.VIDA    import VIDA

from run_base_split import petscVP1Dbasesplit


class petscVP1DbasesplitRK2(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile, runid):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile, runid)
        
        
        # Runge-Kutta collocation points
        self.c1 = 0.5
        
        # Runge-Kutta coefficients
        self.a11 = 0.5
        
        
        # create solution and RHS vector
        self.b  = self.da1.createGlobalVec()
        self.k1 = self.da1.createGlobalVec()
        self.kh = self.da1.createGlobalVec()
        
        # create substep vectors
        self.f1 = self.da1.createGlobalVec()
        self.p1 = self.dax.createGlobalVec()
        self.n1 = self.dax.createGlobalVec()
        self.p1_ext = self.dax.createGlobalVec()
        
        
        self.p1_niter = 0
        
        
    
    def calculate_moments2(self, potential=True, output=True):
        self.fh.copy(self.f1)
        self.f1.axpy(0.5 * self.ht, self.k1)
        
        self.toolbox.compute_density(self.f1, self.n1)
#         self.toolbox.compute_velocity_density(self.f1, self.u1)
#         self.toolbox.compute_velocity_density(self.f2, self.u2)
#         self.toolbox.compute_energy_density(self.f1, self.e1)
#         self.toolbox.compute_energy_density(self.f2, self.e2)
 
        if potential:
            self.poisson_solver.formRHS(self.n1, self.pb)
            self.poisson_ksp.solve(self.pb, self.p1)
            self.p1_niter = self.poisson_ksp.getIterationNumber()
        
        
    
    def calculate_residual2(self):
        self.vlasov_solver.function_mult(self.k1, self.b)
        fnorm = self.b.norm()
        
        self.poisson_solver.function_mult(self.p1, self.n1, self.pb)
        p1norm = self.pb.norm()
        
        return fnorm + p1norm
#         return fnorm
    
    
    def initial_guess2(self):
        self.initial_guess_none()
    
        
    def initial_guess_none(self):
        self.k1.set(0.)
        
        self.f.copy(self.f1)
        self.p.copy(self.p1)
        
    
#     def initial_guess_symplectic2(self):
#         super().initial_guess_symplectic2()
#         self.copy_data_to_x()
#          
#     
#     def initial_guess_symplectic4(self):
#         super().initial_guess_symplectic4()
#         self.copy_data_to_x()
#          
#     
#     def initial_guess_rk4(self):
#         super().initial_guess_rk4()
#         self.copy_data_to_x()
#          
#     
#     def initial_guess_gear(self, itime):
#         super().initial_guess_gear()
#         self.copy_data_to_x()
