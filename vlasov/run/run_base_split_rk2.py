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

from vlasov.run.run_base_split import viVlasov1Dbasesplit


class viVlasov1DbasesplitRK2(viVlasov1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile, runid=None, cfg=None):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile, runid, cfg)
        
        
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
        self.n1 = self.dax.createGlobalVec()
        self.p1_int = self.dax.createGlobalVec()
        self.p1_ext = self.dax.createGlobalVec()
        
        self.h11 = self.da1.createGlobalVec()
        self.h21 = self.da1.createGlobalVec()
        
        self.p1_niter = 0
        
        
    
    def calculate_moments2(self, potential=True, output=True):
        self.fh.copy(self.f1)
        self.f1.axpy(0.5 * self.grid.ht, self.k1)
        
        self.toolbox.compute_density(self.f1, self.n1)
 
        if potential:
            self.poisson_solver.formRHS(self.n1, self.pb)
            self.poisson_ksp.solve(self.pb, self.p1_int)
            self.p1_niter = self.poisson_ksp.getIterationNumber()
            self.toolbox.potential_to_hamiltonian(self.p1_int, self.h11)
        
        
    def calculate_external2(self, itime):
        current_time0 = self.grid.ht*itime
        current_time1 = self.grid.ht*(itime - 1 + self.c1)
        
        # calculate external field
        self.calculate_external(current_time0, self.pc_ext)
        self.calculate_external(current_time1, self.p1_ext)
        
        # copy to Hamiltonian
        self.toolbox.potential_to_hamiltonian(self.p1_ext, self.h21)
        
    
    def calculate_residual2(self):
        self.vlasov_solver.function_mult(self.k1, self.b)
        fnorm = self.b.norm()
        
        self.poisson_solver.function_mult(self.p1_int, self.n1, self.pb)
        p1norm = self.pb.norm()
        
        return fnorm + p1norm
    
    
    def initial_guess2(self):
        self.initial_guess_none()
    
        
    def initial_guess_none(self):
        self.k1.set(0.)
        
        self.fc.copy(self.f1)
        self.pc_int.copy(self.p1_int)
        
    
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
