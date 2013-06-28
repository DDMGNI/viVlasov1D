'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys
import petsc4py
petsc4py.init(sys.argv)

import numpy as np

from petsc4py import PETSc

from vlasov.VIDA    import VIDA
from vlasov.Toolbox import Toolbox

from vlasov.core.config  import Config
from vlasov.data.maxwell import maxwellian

from run_base_split import petscVP1Dbasesplit


class petscVP1Dbasefull(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile, runid):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile, runid)
        
        
        # create VIDA for 2d grid (f, phi and moments)
        self.da2 = VIDA().create(dim=1, dof=self.nv+4,
                                       sizes=[self.nx],
                                       proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                       boundary_type=('periodic'),
                                       stencil_width=2,
                                       stencil_type='box')
        
        # initialise grid
        self.da2.setUniformCoordinates(xmin=0.0,  xmax=L)
        
        # create solution and RHS vector
        self.x  = self.da2.createGlobalVec()
        self.xh = self.da2.createGlobalVec()
        self.xn = self.da2.createGlobalVec()
        self.b  = self.da2.createGlobalVec()
        
        
        # initialise nullspace basis vector for full solution
        # the Poisson equation has a null space of all constant vectors
        # that needs to be removed to avoid jumpy potentials
        self.xn.set(0.)
        x_nvec_arr = self.da2.getGlobalArray(self.xn)
        p_nvec_arr = self.dax.getGlobalArray(self.pn)
        
        x_nvec_arr[:, self.nv] = p_nvec_arr  
#         x_nvec_arr[:, self.nv] = 1.
#         self.x_nvec.normalize()
        
        self.nullspace = PETSc.NullSpace().create(constant=False, vectors=(self.xn,))
        
        
        # create placeholder for solver object
        self.vlasov_poisson_solver = None
        
        
        # copy f, p, and moments to solution vector
        self.copy_data_to_x()
        
    
    
    
    def calculate_residual(self):
        self.vlasov_poisson_solver.function_mult(self.x, self.b)
        norm = self.b.norm()
        
        return norm
    
    
    def calculate_moments(self, potential=True, output=True):
        super().calculate_moments(potential, output)
        
        self.copy_m_to_x()
        
        if potential:
            self.copy_p_to_x()                    # copy potential to solution vector
    
    
    def copy_x_to_data(self):
        self.copy_x_to_f()
        self.copy_x_to_p()
        self.copy_x_to_n()
        self.copy_x_to_u()
        self.copy_x_to_e()
        self.copy_p_to_h()
    
    
    def copy_data_to_x(self):
        self.copy_f_to_x()
        self.copy_p_to_x()
        self.copy_n_to_x()
        self.copy_u_to_x()
        self.copy_e_to_x()
    
    
    def copy_x_to_f(self):
        x_arr = self.da2.getGlobalArray(self.x)
        f_arr = self.da1.getGlobalArray(self.f)
        
        f_arr[:, :] = x_arr[:, 0:self.nv] 
        
    
    def copy_f_to_x(self):
        x_arr = self.da2.getGlobalArray(self.x)
        f_arr = self.da1.getGlobalArray(self.f)
        
        x_arr[:, 0:self.nv] = f_arr[:, :]
        
    
    def copy_x_to_p(self):
        x_arr = self.da2.getGlobalArray(self.x)
        p_arr = self.dax.getGlobalArray(self.p)
        
        p_arr[:] = x_arr[:, self.nv]
        
    
    def copy_p_to_x(self):
        p_arr = self.dax.getGlobalArray(self.p)
        x_arr = self.da2.getGlobalArray(self.x)
        
        x_arr[:, self.nv] = p_arr[:]
        
        
    def copy_x_to_m(self):
        self.copy_x_to_n()
        self.copy_x_to_u()
        self.copy_x_to_e()
 
 
    def copy_m_to_x(self):
        self.copy_n_to_x()
        self.copy_u_to_x()
        self.copy_e_to_x()

    
    def copy_x_to_n(self):
        x_arr = self.da2.getGlobalArray(self.x)
        n_arr = self.dax.getGlobalArray(self.n)
        
        n_arr[:] = x_arr[:, self.nv+1]
        
    
    def copy_n_to_x(self):
        n_arr = self.dax.getGlobalArray(self.n)
        x_arr = self.da2.getGlobalArray(self.x)
        
        x_arr[:, self.nv+1] = n_arr[:]
        
        
    def copy_x_to_u(self):
        x_arr = self.da2.getGlobalArray(self.x)
        u_arr = self.dax.getGlobalArray(self.u)
        
        u_arr[:] = x_arr[:, self.nv+2]
        
    
    def copy_u_to_x(self):
        u_arr = self.dax.getGlobalArray(self.u)
        x_arr = self.da2.getGlobalArray(self.x)
        
        x_arr[:, self.nv+2] = u_arr[:]
        
        
    def copy_x_to_e(self):
        x_arr = self.da2.getGlobalArray(self.x)
        e_arr = self.dax.getGlobalArray(self.e)
        
        e_arr[:] = x_arr[:, self.nv+3]
        
    
    def copy_e_to_x(self):
        e_arr = self.dax.getGlobalArray(self.e)
        x_arr = self.da2.getGlobalArray(self.x)
        
        x_arr[:, self.nv+3] = e_arr[:]
        
        
    def initial_guess_symplectic2(self):
        super().initial_guess_symplectic2()
        self.copy_data_to_x()
         
    
    def initial_guess_symplectic4(self):
        super().initial_guess_symplectic4()
        self.copy_data_to_x()
         
    
    def initial_guess_rk4(self):
        super().initial_guess_rk4()
        self.copy_data_to_x()
         
    
    def initial_guess_gear(self, itime):
        super().initial_guess_gear()
        self.copy_data_to_x()
