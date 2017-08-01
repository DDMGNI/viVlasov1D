'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys
import petsc4py
petsc4py.init(sys.argv)

import numpy as np

from petsc4py import PETSc

from vlasov.toolbox.Arakawa import Arakawa
from vlasov.toolbox.VIDA    import VIDA

from run_base_split import viVlasov1Dbasesplit


class viVlasov1DbasesplitDB(viVlasov1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile, runid=None, cfg=None):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile, runid, cfg)
        
        
        # create VIDA for 2d grid (f, phi and moments)
        self.da2 = VIDA().create(dim=2, dof=2,
                                       sizes=[self.grid.nx, self.grid.nv],
                                       proc_sizes=[1, PETSc.COMM_WORLD.getSize()],
                                       boundary_type=('periodic', 'ghosted'),
                                       stencil_width=self.grid.stencil,
                                       stencil_type='box')
        
        # initialise grid
        self.da2.setUniformCoordinates(xmin=self.grid.xMin(),  xmax=self.grid.xMax(), ymin=self.grid.vMin(), ymax=self.grid.vMax())
        
        # create solution and RHS vector
        self.b  = self.da2.createGlobalVec()
        self.k  = self.da2.createGlobalVec()
        self.kh = self.da2.createGlobalVec()
        
        # create substep vectors
        self.gc = self.da1.createGlobalVec()
        self.gh = self.da1.createGlobalVec()
        
        # create Arakawa object
        self.arakawa = Arakawa(self.da1, self.grid)

        
    
    def make_history_db(self, update_solver=True):
        self.make_history(update_solver=False)
        
        if update_solver and self.vlasov_solver != None:
            self.vlasov_solver.update_history_db(self.fc, self.gc)
        
        
    def calculate_moments_db(self, potential=True, output=True):
        k_arr = self.da2.getGlobalArray(self.k)
        f_arr = getGlobalArray(self.da1, self.fc)
        g_arr = getGlobalArray(self.da1, self.gc)
        
        f_arr[:, :] = k_arr[:, :, 0]
        g_arr[:, :] = k_arr[:, :, 1]
        
        self.calculate_moments(potential, output)
        
        
    def calculate_residual_db(self):
        self.vlasov_solver.function_mult(self.k, self.b)
        fnorm = self.b.norm()
        
        self.poisson_solver.function_mult(self.pc_int, self.N, self.pb)
        pnorm = self.pb.norm()
        
        return fnorm + pnorm
    
    
#     def initial_guess_db(self):
#         self.initial_guess_none()
#         self.calculate_moments_db(output=False)
    
        
#     def initial_guess_none(self):
#         self.k.set(0.)
        