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

from vlasov.vi.PETScSimpleNLFunctionCollT   import PETScFunction
from vlasov.vi.PETScSimpleNLJacobianCollT   import PETScJacobian
from vlasov.vi.PETScSimpleNLMFJacobianCollT import PETScJacobianMatrixFree

from petscvp1d import petscVP1Dbase


class petscVP1D(petscVP1Dbase):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    
    Solve without initial guess from linearised scheme, use last timestep instead.
    Run matrix-based approximate Jacobian without collisions.
    Run matrix-based/matrix-free predictor only to low accuracy
    Improve on that with GMRES only solver
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        # initialise parent object
        petscVP1Dbase.__init__(self, cfgfile)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_atol',    self.cfg['solver']['petsc_ksp_atol'])
        OptDB.setValue('ksp_rtol',    self.cfg['solver']['petsc_ksp_rtol'])
        OptDB.setValue('ksp_stol',    self.cfg['solver']['petsc_ksp_stol'])
        OptDB.setValue('ksp_max_it',  self.cfg['solver']['petsc_ksp_max_iter'])
        
        OptDB.setValue('ksp_monitor',  '')
        OptDB.setValue('snes_monitor', '')
        
#        OptDB.setValue('log_info',    '')
#        OptDB.setValue('log_summary', '')
        
        
        self.ksp_atol  = 1E-13
        self.snes_atol = 1E-10
        
        
        # create residual vector
        self.F  = self.da2.createGlobalVec()
        
        # create Jacobian, Function, and linear Matrix objects
        self.petsc_function = PETScFunction(self.da1, self.da2, self.dax, 
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.charge, coll_freq=self.coll_freq)
        
        self.petsc_jacobian_mf = PETScJacobianMatrixFree(self.da1, self.da2, self.dax,
                                                         self.h0, self.vGrid,
                                                         self.nx, self.nv, self.ht, self.hx, self.hv,
                                                         self.charge, coll_freq=self.coll_freq)
        
        self.petsc_jacobian = PETScJacobian(self.da1, self.da2, self.dax,
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.charge)
        
        
        # initialise Jacobian
        self.J = self.da2.createMat()
        self.J.setOption(self.J.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.J.setUp()

        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.x.getSizes(), self.F.getSizes()], 
                                            context=self.petsc_jacobian_mf,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
        
        
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
        
        
        # create nonlinear solver
        self.snes_nonlinear = PETSc.SNES().create()
        self.snes_nonlinear.setType('ksponly')
        self.snes_nonlinear.setFunction(self.petsc_function.snes_mult, self.F)
        self.snes_nonlinear.setJacobian(self.updateJacobian, self.Jmf, self.J)
        self.snes_nonlinear.setFromOptions()
        self.snes_nonlinear.getKSP().setType('gmres')
        self.snes_nonlinear.getKSP().getPC().setType('lu')
        self.snes_nonlinear.getKSP().getPC().setFactorSolverPackage('mumps')
        
        
        OptDB.setValue('ksp_atol', self.ksp_atol)
        OptDB.setValue('ksp_rtol', 1E-5)
        
        # create nonlinear solver
        self.snes_backup = PETSc.SNES().create()
        self.snes_backup.setType('ksponly')
        self.snes_backup.setFunction(self.petsc_function.snes_mult, self.F)
        self.snes_backup.setJacobian(self.updateJacobianMF, self.Jmf)
        self.snes_backup.setFromOptions()
        self.snes_backup.getKSP().setType('gmres')
        self.snes_backup.getKSP().getPC().setType('none')
        
        
        
        # calculate initial potential
        self.calculate_potential()
        
        # copy external potential
        self.petsc_jacobian_mf.update_external(self.p_ext)
        self.petsc_jacobian.update_external(self.p_ext)
        self.petsc_function.update_external(self.p_ext)
        
        # update solution history
        self.petsc_jacobian_mf.update_history(self.f, self.h1)
        self.petsc_jacobian.update_history(self.f, self.h1)
        self.petsc_function.update_history(self.f, self.h1, self.p)
        
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
    
        
    def updateJacobian(self, snes, X, J, P):
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
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # calculate external field and copy to matrix, jacobian and function
            self.calculate_external(current_time)
            self.petsc_jacobian_mf.update_external(self.p_ext)
            self.petsc_jacobian.update_external(self.p_ext)
            self.petsc_function.update_external(self.p_ext)
            
            
            # nonlinear solve
            i = 0
            while True:
                i+=1
                
                self.snes_nonlinear.solve(None, self.x)
                
                self.petsc_function.mult(self.x, self.F)
                fnorm = self.F.norm()
                
                # output some solver info
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlin Solver:  %5i iterations,   funcnorm = %24.16E" % (i, fnorm) )
                
                if fnorm < self.cfg['solver']['petsc_snes_atol'] or i >= 20:
                    break
           
           
            # backup solve
            i = 0
            while True:
                i+=1
                
                self.snes_backup.solve(None, self.x)
                
                self.petsc_function.mult(self.x, self.F)
                fnorm = self.F.norm()
                
                # output some solver info
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Backup Solver:  %5i iterations,   funcnorm = %24.16E" % (i, fnorm) )
                
                if fnorm < self.snes_atol or i >= self.cfg['solver']['petsc_snes_max_iter']:
                    break
           
           
            # update data vectors
            self.copy_x_to_f()
            self.copy_x_to_p()
            self.copy_p_to_h()
            
            # update history
            self.petsc_jacobian_mf.update_history(self.f, self.h1)
            self.petsc_jacobian.update_history(self.f, self.h1)
            self.petsc_function.update_history(self.f, self.h1, self.p)
            
            # save to hdf5
            self.save_to_hdf5(itime)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1D(args.runfile)
    petscvp.run()
    
