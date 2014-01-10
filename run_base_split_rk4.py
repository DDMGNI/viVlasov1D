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


class petscVP1DbasesplitRK4(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile, runid):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile, runid)
        
        
        # Runge-Kutta collocation points
        self.c1 = 0.5 - 0.5 / np.sqrt(3.)
        self.c2 = 0.5 + 0.5 / np.sqrt(3.)
#         self.c1 = 0.
#         self.c2 = 1.
#         self.c1 = 0.
#         self.c2 = 0.5
        
        # Runge-Kutta weights
        self.b1 = 0.5
        self.b2 = 0.5
#         self.b1 = 0.5
#         self.b2 = 0.5
#         self.b1 = 0.
#         self.b2 = 1.
        
        # Runge-Kutta coefficients
        self.a11 = 0.25
        self.a12 = 0.25 - 0.5 / np.sqrt(3.) 
        self.a21 = 0.25 + 0.5 / np.sqrt(3.) 
        self.a22 = 0.25
#         self.a11 = 0.
#         self.a12 = 0.
#         self.a21 = 0.5
#         self.a22 = 0.5
#         self.a11 = 0.
#         self.a12 = 0.
#         self.a21 = 0.5
#         self.a22 = 0.
        
        
        # create VIDA for 2d grid (f, phi and moments)
        self.da2 = VIDA().create(dim=2, dof=2,
                                       sizes=[self.nx, self.nv],
                                       proc_sizes=[PETSc.COMM_WORLD.getSize(), 1],
                                       boundary_type=('periodic', 'none'),
                                       stencil_width=2,
                                       stencil_type='box')
        
        # initialise grid
        self.da2.setUniformCoordinates(xmin=0.0,  xmax=self.Lx, ymin=self.vMin, ymax=self.vMax)
        
        # create solution and RHS vector
        self.b  = self.da2.createGlobalVec()
        self.k  = self.da2.createGlobalVec()
        self.kh = self.da2.createGlobalVec()
        
        # create substep vectors
        self.k1 = self.da1.createGlobalVec()
        self.k2 = self.da1.createGlobalVec()
        self.f1 = self.da1.createGlobalVec()
        self.f2 = self.da1.createGlobalVec()
        self.p1 = self.dax.createGlobalVec()
        self.p2 = self.dax.createGlobalVec()
        self.n1 = self.dax.createGlobalVec()
        self.n2 = self.dax.createGlobalVec()
        self.p1_ext = self.dax.createGlobalVec()
        self.p2_ext = self.dax.createGlobalVec()
        
        
        self.p1_niter = 0
        self.p2_niter = 0
        
        
    
    def calculate_moments4(self, potential=True, output=True):
        k_arr  = self.da2.getGlobalArray(self.k)
        k1_arr = self.da1.getGlobalArray(self.k1)
        k2_arr = self.da1.getGlobalArray(self.k2)
        
        k1_arr[:, :] = k_arr[:, :, 0]
        k2_arr[:, :] = k_arr[:, :, 1]
        
        self.fh.copy(self.f1)
        self.fh.copy(self.f2)
        
        self.f1.axpy(self.ht * self.a11, self.k1)
        self.f1.axpy(self.ht * self.a12, self.k2)
        self.f2.axpy(self.ht * self.a21, self.k1)
        self.f2.axpy(self.ht * self.a22, self.k2)
        
        self.toolbox.compute_density(self.f1, self.n1)
        self.toolbox.compute_density(self.f2, self.n2)
#         self.toolbox.compute_velocity_density(self.f1, self.u1)
#         self.toolbox.compute_velocity_density(self.f2, self.u2)
#         self.toolbox.compute_energy_density(self.f1, self.e1)
#         self.toolbox.compute_energy_density(self.f2, self.e2)
 
        if potential:
            self.poisson_solver.formRHS(self.n1, self.pb)
            self.poisson_ksp.solve(self.pb, self.p1)
            self.p1_niter = self.poisson_ksp.getIterationNumber()
            
            self.poisson_solver.formRHS(self.n2, self.pb)
            self.poisson_ksp.solve(self.pb, self.p2)
            self.p2_niter = self.poisson_ksp.getIterationNumber()
        
        
    
    def calculate_residual4(self):
        self.vlasov_solver.function_mult(self.k, self.b)
        fnorm = self.b.norm()
        
        self.poisson_solver.function_mult(self.p1, self.n1, self.pb)
        p1norm = self.pb.norm()
        
        self.poisson_solver.function_mult(self.p2, self.n2, self.pb)
        p2norm = self.pb.norm()
        
        return fnorm + p1norm + p2norm
    
    
    def initial_guess4(self):
        self.initial_guess_none()
        self.calculate_moments4(output=False)
    
        
    def initial_guess_none(self):
        self.k.set(0.)
        
    
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
