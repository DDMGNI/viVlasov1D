'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time

import numpy as np

from vlasov.predictor.PETScPoissonMatrix import PETScPoissonMatrix

from vlasov.vi.PETScSimpleMatrixCollT       import PETScMatrix
from vlasov.vi.PETScSimpleNLFunctionCollT   import PETScFunction
from vlasov.vi.PETScSimpleNLJacobianCollT   import PETScJacobian
from vlasov.vi.PETScSimpleNLMFJacobianCollT import PETScJacobianMatrixFree


from petscvp1d import petscVP1Dbase


class petscVP1D(petscVP1Dbase):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    
    Run matrix-based approximate Jacobian with just diffusion operator.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        # initialise parent object
        petscVP1Dbase.__init__(self, cfgfile)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
#        OptDB.setValue('snes_atol',   self.cfg['solver']['petsc_snes_atol'])
#        OptDB.setValue('snes_rtol',   self.cfg['solver']['petsc_snes_rtol'])
#        OptDB.setValue('snes_stol',   self.cfg['solver']['petsc_snes_stol'])
#        OptDB.setValue('snes_max_it', 20)

#        OptDB.setValue('ksp_atol',    self.cfg['solver']['petsc_ksp_atol'])
#        OptDB.setValue('ksp_rtol',    self.cfg['solver']['petsc_ksp_rtol'])
#        OptDB.setValue('ksp_stol',    self.cfg['solver']['petsc_ksp_stol'])
#        OptDB.setValue('ksp_max_it',  100)
        
        OptDB.setValue('snes_lag_preconditioner', 5)
        
        OptDB.setValue('ksp_monitor',  '')
        OptDB.setValue('snes_monitor', '')
        
#        OptDB.setValue('log_info',    '')
#        OptDB.setValue('log_summary', '')


        
        # create residual vector
        self.F  = self.da2.createGlobalVec()
        
        # create Jacobian, Function, and linear Matrix objects
        self.petsc_jacobian_mf = PETScJacobianMatrixFree(self.da1, self.da2, self.dax,
                                                         self.h0, self.vGrid,
                                                         self.nx, self.nv, self.ht, self.hx, self.hv,
                                                         self.charge, coll_freq=self.coll_freq)
        
        self.petsc_jacobian = PETScJacobian(
                                            self.da1, self.da2, self.dax,
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.charge, coll_freq=self.coll_freq)
        
        self.petsc_function = PETScFunction(
                                            self.da1, self.da2, self.dax, 
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.charge, coll_freq=self.coll_freq)
        
        self.petsc_matrix = PETScMatrix(self.da1, self.da2, self.dax,
                                        self.h0, self.vGrid,
                                        self.nx, self.nv, self.ht, self.hx, self.hv,
                                        self.charge)
        
        
        # initialise matrix
        self.A = self.da2.createMat()
        self.A.setOption(self.A.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.A.setUp()

        # initialise Jacobian
        self.J = self.da2.createMat()
        self.J.setOption(self.J.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.J.setUp()

        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.x.getSizes(), self.F.getSizes()], 
                                            context=self.petsc_jacobian_mf,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
        
        
        OptDB.setValue('snes_atol',   1E-7)
        OptDB.setValue('snes_rtol',   1E-8)
        OptDB.setValue('ksp_atol',    1E-12)
        OptDB.setValue('ksp_rtol',    1E-11)
        
        # create linear predictor
        self.snes_linear = PETSc.SNES().create()
        self.snes_linear.setType('ksponly')
        self.snes_linear.setFunction(self.petsc_matrix.snes_mult, self.F)
        self.snes_linear.setJacobian(self.updateMatrix, self.A)
        self.snes_linear.setFromOptions()
        self.snes_linear.getKSP().setType('preonly')
        self.snes_linear.getKSP().getPC().setType('lu')
        self.snes_linear.getKSP().getPC().setFactorSolverPackage('mumps')

        
        # create nonlinear solver
        self.snes_nonlinear = PETSc.SNES().create()
        self.snes_nonlinear.setFunction(self.petsc_function.snes_mult, self.F)
        self.snes_nonlinear.setJacobian(self.updateJacobian, self.J)
        self.snes_nonlinear.setFromOptions()
        self.snes_nonlinear.getKSP().setType('gmres')
        self.snes_nonlinear.getKSP().getPC().setType('lu')
        self.snes_nonlinear.getKSP().getPC().setFactorSolverPackage('mumps')
        

        OptDB.setValue('snes_atol',   1E-8)
        OptDB.setValue('snes_rtol',   1E-11)
        OptDB.setValue('snes_stol',   1E-12)
        OptDB.setValue('snes_max_it', self.cfg['solver']['petsc_snes_max_iter'])
        OptDB.setValue('ksp_atol',    1E-10)
        OptDB.setValue('ksp_rtol',    1E-7)
        OptDB.setValue('ksp_stol',    1E-14)
        OptDB.setValue('ksp_max_it',  self.cfg['solver']['petsc_ksp_max_iter'])
        
        
#        OptDB.setValue('sub_pc_type',  'lu')
#        OptDB.setValue('sub_pc_factor_mat_solver_package',  'mumps')
        OptDB.setValue('ksp_gmres_restart',  100)


        # create nonlinear solver with collisions
        self.snes_collisions = PETSc.SNES().create()
        self.snes_collisions.setFunction(self.petsc_function.snes_mult, self.F)
        self.snes_collisions.setJacobian(self.updateJacobianBoth, self.Jmf, self.J)
        self.snes_collisions.setFromOptions()
        self.snes_collisions.getKSP().setType('gmres')
#        self.snes_collisions.getKSP().getPC().setType('asm')
        self.snes_collisions.getKSP().getPC().setType('lu')
        self.snes_collisions.getKSP().getPC().setFactorSolverPackage('mumps')
        
        
#        OptDB.setValue('snes_atol',   self.cfg['solver']['petsc_snes_atol'])
#        OptDB.setValue('snes_rtol',   self.cfg['solver']['petsc_snes_rtol'])
#        OptDB.setValue('snes_stol',   self.cfg['solver']['petsc_snes_stol'])
#        OptDB.setValue('ksp_atol',    self.cfg['solver']['petsc_ksp_atol'])
#        OptDB.setValue('ksp_rtol',    self.cfg['solver']['petsc_ksp_rtol'])
#        OptDB.setValue('ksp_stol',    self.cfg['solver']['petsc_ksp_stol'])
        
        OptDB.setValue('snes_atol',   1E-11)
        OptDB.setValue('ksp_atol',    1E-11)
        
        # create backup solver
        self.snes_backup = PETSc.SNES().create()
        self.snes_backup.setFunction(self.petsc_function.snes_mult, self.F)
        self.snes_backup.setJacobian(self.updateJacobianMF, self.Jmf)
        self.snes_backup.setFromOptions()
        self.snes_backup.getKSP().setType('gmres')
        self.snes_backup.getKSP().getPC().setType('none')
        
        
        # create Poisson object
        self.poisson_mat = PETScPoissonMatrix(self.da1, self.dax, 
                                              self.nx, self.nv, self.hx, self.hv,
                                              self.charge)
        
        # initialise Poisson matrix
        self.poisson_A = self.dax.createMat()
        self.poisson_A.setOption(self.poisson_A.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.poisson_A.setUp()
        
        # create linear Poisson solver
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_A)
        self.poisson_ksp.setType('preonly')
        self.poisson_ksp.getPC().setType('lu')
        self.poisson_ksp.getPC().setFactorSolverPackage('mumps')
        
        
        # calculate initial potential
        self.calculate_potential()
        
        # copy external potential
        self.petsc_jacobian_mf.update_external(self.p_ext)
        self.petsc_jacobian.update_external(self.p_ext)
        self.petsc_function.update_external(self.p_ext)
        self.petsc_matrix.update_external(self.p_ext)
        
        # update solution history
        self.petsc_jacobian_mf.update_history(self.f, self.h1)
        self.petsc_jacobian.update_history(self.f, self.h1)
        self.petsc_function.update_history(self.f, self.h1, self.p)
        self.petsc_matrix.update_history(self.f, self.h1)
        
        # save to hdf5
        self.hdf5_viewer.HDF5SetTimestep(0)
        self.save_hdf5_vectors()
        
        
    def calculate_potential(self):
        
        self.poisson_mat.formMat(self.poisson_A)
        self.poisson_mat.formRHS(self.f, self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        phisum = self.p.sum()
        
        self.copy_p_to_x()
        self.copy_p_to_h()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Poisson:  %5i iterations,   residual = %24.16E" % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
            print("                                   sum(phi) = %24.16E" % (phisum))
    
        
    def updateMatrix(self, snes, X, J, P):
        self.petsc_matrix.formMat(J)
    
    
    def updateJacobian(self, snes, X, J, P):
        self.petsc_jacobian.update_previous(X)
        self.petsc_jacobian.formMat(J)
    
    
    def updateJacobianBoth(self, snes, X, J, P):
        self.petsc_jacobian_mf.update_previous_X(X)
        self.petsc_jacobian.update_previous(X)
        self.petsc_jacobian.formMat(P)
    
    
    def updateJacobianMF(self, snes, X, J, P):
        self.petsc_jacobian_mf.update_previous_X(X)
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                self.time.setValue(0, current_time)
            
            # calculate external field and copy to matrix, jacobian and function
            self.calculate_external(current_time)
            self.petsc_jacobian_mf.update_external(self.p_ext)
            self.petsc_jacobian.update_external(self.p_ext)
            self.petsc_function.update_external(self.p_ext)
            self.petsc_matrix.update_external(self.p_ext)
            
            
#            # calculate initial guess
#            self.snes_linear.solve(None, self.x)
#            
#            if PETSc.COMM_WORLD.getRank() == 0:
#                print()
#                print("  Linear Precon:  %5i iterations,   funcnorm = %24.16E" % (self.snes_linear.getIterationNumber(), self.snes_linear.getFunctionNorm()) )
#                print()
#            
#            
#            # nonlinear solve
#            self.snes_nonlinear.solve(None, self.x)
#           
#            if PETSc.COMM_WORLD.getRank() == 0:
#                print()
#                print("  Nonlin Precon:  %5i iterations,   funcnorm = %24.16E" % (self.snes_nonlinear.getIterationNumber(), self.snes_nonlinear.getFunctionNorm()) )
#                print()
#            
#            if self.snes_backup.getConvergedReason() < 0:
#                if PETSc.COMM_WORLD.getRank() == 0:
#                    print()
#                    print("Solver not converging...   %i" % (self.snes_nonlinear.getConvergedReason()))
#                    print()
           
           
            # collisional solve
            self.snes_collisions.solve(None, self.x)
           
            # output some solver info
            if PETSc.COMM_WORLD.getRank() == 0:
                print()
                print("  Nonlin Solver:  %5i iterations,   funcnorm = %24.16E" % (self.snes_collisions.getIterationNumber(), self.snes_collisions.getFunctionNorm()) )
                print()
            
            if self.snes_nonlinear.getConvergedReason() < 0:
                if PETSc.COMM_WORLD.getRank() == 0:
                    print()
                    print("Solver not converging...   %i" % (self.snes_collisions.getConvergedReason()))
                    print()
           
            
            # backup solve
            self.snes_backup.solve(None, self.x)
           
            if PETSc.COMM_WORLD.getRank() == 0:
                print()
                print("  Backup Solver:  %5i iterations,   funcnorm = %24.16E" % (self.snes_backup.getIterationNumber(), self.snes_backup.getFunctionNorm()) )
                print()
            
            if self.snes_backup.getConvergedReason() < 0:
                if PETSc.COMM_WORLD.getRank() == 0:
                    print()
                    print("Solver not converging...   %i" % (self.snes_backup.getConvergedReason()))
                    print()
           
           
            # update data vectors
            self.copy_x_to_f()
            self.copy_x_to_p()
            self.copy_p_to_h()
            
            # update history
            self.petsc_jacobian_mf.update_history(self.f, self.h1)
            self.petsc_jacobian.update_history(self.f, self.h1)
            self.petsc_function.update_history(self.f, self.h1, self.p)
            self.petsc_matrix.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1D(args.runfile)
    petscvp.run()
#    petscvp.check_jacobian()
    
