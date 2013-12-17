'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys
import petsc4py
petsc4py.init(sys.argv)

# import numpy as np

from petsc4py import PETSc

from vlasov.toolbox.VIDA    import VIDA

from run_base_split import petscVP1Dbasesplit


class petscVP1DbasesplitRK4(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile, runid):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile, runid)
        
        
        # create VIDA for 2d grid (f, phi and moments)
        self.da2 = VIDA().create(dim=2, dof=2,
                                       sizes=[self.nx, self.nv],
                                       proc_sizes=[PETSc.COMM_WORLD.getSize(), 1],
                                       boundary_type=('periodic', 'none'),
                                       stencil_width=2,
                                       stencil_type='box')
        
        # initialise grid
        self.da2.setUniformCoordinates(xmin=0.0,  xmax=self.Lx, ymin=-self.vMax, ymax=+self.vMax)
        
        # create solution and RHS vector
        self.b  = self.da2.createGlobalVec()
        self.x  = self.da2.createGlobalVec()
        self.xh = self.da2.createGlobalVec()
        
        # create substep vectors
        self.f1 = self.da1.createGlobalVec()
        self.f2 = self.da1.createGlobalVec()
        self.p1 = self.dax.createGlobalVec()
        self.p2 = self.dax.createGlobalVec()
        self.n1 = self.dax.createGlobalVec()
        self.n2 = self.dax.createGlobalVec()
        self.p1_ext = self.dax.createGlobalVec()
        self.p2_ext = self.dax.createGlobalVec()
        
        
    
    def calculate_moments4(self, potential=True, output=True):
        x_arr  = self.da2.getGlobalArray(self.x)
        f1_arr = self.da1.getGlobalArray(self.f1)
        f2_arr = self.da1.getGlobalArray(self.f2)
        
        f1_arr[:, :] = x_arr[:, :, 0]
        f2_arr[:, :] = x_arr[:, :, 1]
        
        self.fh.copy(self.f)
        self.f.axpy(0.5, self.f1)
        self.f.axpy(0.5, self.f2)
        
#         super().calculate_moments(potential=potential, output=output)
        self.calculate_moments(potential=potential, output=output)
        
        self.toolbox.compute_density(self.f1, self.n1)
        self.toolbox.compute_density(self.f2, self.n2)
#         self.toolbox.compute_velocity_density(self.f1, self.u1)
#         self.toolbox.compute_velocity_density(self.f2, self.u2)
#         self.toolbox.compute_energy_density(self.f1, self.e1)
#         self.toolbox.compute_energy_density(self.f2, self.e2)
 
        if potential:
            self.poisson_solver.formRHS(self.n1, self.pb)
            self.poisson_ksp.solve(self.pb, self.p1)
        
            self.poisson_solver.formRHS(self.n2, self.pb)
            self.poisson_ksp.solve(self.pb, self.p2)
        
        
    
    def calculate_residual4(self):
        self.vlasov_solver.function_mult(self.x, self.b)
        fnorm = self.b.norm()
        
        self.poisson_solver.function_mult(self.p1, self.n1, self.pb)
        p1norm = self.pb.norm()
        
        self.poisson_solver.function_mult(self.p2, self.n2, self.pb)
        p2norm = self.pb.norm()
        
        return fnorm + p1norm + p2norm
    
    
    def initial_guess4(self):
        self.initial_guess_none()
    
        
    def initial_guess_none(self):
        self.x.set(0.)
        self.f1.set(0.)
        self.f2.set(0.)
        self.p1.set(0.)
        self.p2.set(0.)
        
    
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
