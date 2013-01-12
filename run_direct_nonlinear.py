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
from vlasov.vi.PETScSimpleNLFunction     import PETScFunction
from vlasov.vi.PETScSimpleNLJacobian     import PETScJacobian

from petscvp1d import petscVP1Dbase


class petscVP1D(petscVP1Dbase):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        # initialise parent object
        petscVP1Dbase.__init__(self, cfgfile)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        OptDB.setValue('snes_rtol', 1E-20)
#        OptDB.setValue('snes_atol', 1E-12)
#        OptDB.setValue('snes_atol', 1E-1)
#        OptDB.setValue('snes_atol', 1E-3)
#        OptDB.setValue('snes_atol', 1E-4)
#        OptDB.setValue('snes_atol', 1E-5)
#        OptDB.setValue('snes_atol', 1E-6)
#        OptDB.setValue('snes_atol', 1E-10)
        OptDB.setValue('snes_atol', 1E-11)
#        OptDB.setValue('snes_atol', 1E-12)
        OptDB.setValue('snes_stol', 1E-14)
        
#        OptDB.setValue('ksp_monitor', '')
        OptDB.setValue('snes_monitor', '')
#        OptDB.setValue('log_info', '')
#        OptDB.setValue('log_summary', '')


        
        # create residual vector
        self.F  = self.da2.createGlobalVec()
        
        # create Jacobian and Function objects
        self.petsc_jacobian = PETScJacobian(self.da1, self.da2, self.dax,
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.poisson, self.alpha)
        
        self.petsc_function = PETScFunction(self.da1, self.da2, self.dax, 
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.poisson, self.alpha)
        
        
        # initialise Jacobian
        self.J = self.da2.createMat()
        self.J.setOption(self.J.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.J.setUp()
        
        # create nonlinear solver
        self.snes = PETSc.SNES().create()
        self.snes.setFunction(self.petsc_function.snes_mult, self.F)
        self.snes.setJacobian(self.updateJacobian, self.J)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('preonly')
        self.snes.getKSP().getPC().setType('lu')
        self.snes.getKSP().getPC().setFactorSolverPackage('superlu_dist')
#        self.snes.getKSP().getPC().setFactorSolverPackage('mumps')
        
        
        # create Poisson object
        self.poisson_mat = PETScPoissonMatrix(self.da1, self.dax, 
                                              self.nx, self.nv, self.hx, self.hv,
                                              self.poisson)
        
        # initialise Poisson matrix
        self.poisson_A = self.dax.createMat()
        self.poisson_A.setOption(self.poisson_A.Option.NEW_NONZERO_ALLOCATION_ERR, False)
#        self.poisson_A.setType('mpiaij')
        self.poisson_A.setUp()
        
        # create linear Poisson solver
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_A)
        self.poisson_ksp.setType('preonly')
        self.poisson_ksp.getPC().setType('lu')
        self.poisson_ksp.getPC().setFactorSolverPackage('superlu_dist')
#        self.poisson_ksp.getPC().setFactorSolverPackage('mumps')
        
#        self.poisson_nsp = PETSc.NullSpace().create(constant=True)
#        self.poisson_ksp.setNullSpace(self.poisson_nsp)        
        
        
        # calculate initial potential
        self.calculate_potential()
        
        # update solution history
        self.petsc_function.update_history(self.f, self.h1, self.p)
        self.petsc_jacobian.update_history(self.f, self.h1)
        
        # save to hdf5
        self.hdf5_viewer.HDF5SetTimestep(0)
        self.save_hdf5_vectors()
        
        
    def calculate_potential(self):
        
        self.poisson_mat.formMat(self.poisson_A)
        self.poisson_mat.formRHS(self.f, self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        phisum = self.p.sum()
        
#        self.remove_average_from_potential()
        
        self.copy_p_to_x()
        self.copy_p_to_h()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Poisson:  %5i iterations,   residual = %24.16E" % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
            print("                                   sum(phi) = %24.16E" % (phisum))
    
        
    
    def updateJacobian(self, snes, X, J, P):
        self.petsc_jacobian.update_previous(X)
        self.petsc_jacobian.formMat(J)
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                print
                self.time.setValue(0, self.ht*itime)
            
            # calculate initial guess for distribution function
#            self.initial_guess()
            
            # solve
            self.snes.solve(None, self.x)
            
            # output some solver info
            if PETSc.COMM_WORLD.getRank() == 0:
                print()
                print("  Nonlin Solver:  %5i iterations,   funcnorm = %24.16E" % (self.snes.getIterationNumber(), self.snes.getFunctionNorm()) )
                print()
            
            if self.snes.getConvergedReason() < 0:
                if PETSc.COMM_WORLD.getRank() == 0:
                    print()
                    print("Solver not converging... quitting!")
                    print()
                exit()
           
           
            # update data vectors
            self.copy_x_to_f()
            self.copy_x_to_p()
            
#            self.remove_average_from_potential()
        
            self.copy_p_to_x()
            self.copy_p_to_h()
            
            # update history
            self.petsc_function.update_history(self.f, self.h1, self.p)
            self.petsc_jacobian.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
            
#            # some solver output
            phisum = self.p.sum()
#            
#            
            if PETSc.COMM_WORLD.getRank() == 0:
#                print("     Solver")
                print("     sum(phi) = %24.16E" % (phisum))
##                print("     Solver:   %5i iterations,   sum(phi) = %24.16E" % (phisum))
##                print("                               res(solver)  = %24.16E" % (res_solver))
##                print("                               res(vlasov)  = %24.16E" % (res_vlasov))
##                print("                               res(poisson) = %24.16E" % (res_poisson))
##                print
                
            
            

    def check_jacobian(self):
        
        (xs, xe), = self.da1.getRanges()
        
        eps = 1.E-7
        
        # calculate initial guess
#        self.calculate_initial_guess()
        
        # update previous iteration
        self.petsc_jacobian.update_previous(self.x)
        
        # calculate jacobian
        self.petsc_jacobian.formMat(self.J)
        
        # create working vectors
        Jx  = self.da2.createGlobalVec()
        dJ  = self.da2.createGlobalVec()
        ex  = self.da2.createGlobalVec()
        dx  = self.da2.createGlobalVec()
        dF  = self.da2.createGlobalVec()
        Fxm = self.da2.createGlobalVec()
        Fxp = self.da2.createGlobalVec()
        
        
#        sx = -1
        sx =  0
#        sx = +1
        
        nfield=self.nv+1
        
        for ifield in range(0, nfield):
            for ix in range(xs, xe):
                for tfield in range(0, nfield):
                    
                    # compute ex
                    ex_arr = self.da2.getVecArray(ex)
                    ex_arr[:] = 0.
                    ex_arr[(ix+sx) % self.nx, ifield] = 1.
                    
                    
                    # compute J.e
                    self.J.mult(ex, dJ)
                    
                    dJ_arr = self.da2.getVecArray(dJ)
                    Jx_arr = self.da2.getVecArray(Jx)
                    Jx_arr[ix, tfield] = dJ_arr[ix, tfield]
                    
                    
                    # compute F(x - eps ex)
                    self.x.copy(dx)
                    dx_arr = self.da2.getVecArray(dx)
                    dx_arr[(ix+sx) % self.nx, ifield] -= eps
                    
                    self.petsc_function.mult(dx, Fxm)
                    
                    
                    # compute F(x + eps ex)
                    self.x.copy(dx)
                    dx_arr = self.da2.getVecArray(dx)
                    dx_arr[(ix+sx) % self.nx, ifield] += eps
                    
                    self.petsc_function.mult(dx, Fxp)
                    
                    
                    # compute dF = [F(x + eps ex) - F(x - eps ex)] / (2 eps)
                    Fxm_arr = self.da2.getVecArray(Fxm)
                    Fxp_arr = self.da2.getVecArray(Fxp)
                    dF_arr  = self.da2.getVecArray(dF)
                    
                    dF_arr[ix, tfield] = ( Fxp_arr[ix, tfield] - Fxm_arr[ix, tfield] ) / (2. * eps)
                        
            
            diff = np.zeros(nfield)
            
            for tfield in range(0,nfield):
#                print()
#                print("Fields: (%5i, %5i)" % (ifield, tfield))
#                print()
                
                Jx_arr = self.da2.getVecArray(Jx)[...][:, tfield]
                dF_arr = self.da2.getVecArray(dF)[...][:, tfield]
                
                
                print("Jacobian:")
                print(Jx_arr)
                print()
                
                print("[F(x+dx) - F(x-dx)] / [2 eps]:")
                print(dF_arr)
                print()
#                
#                print("Difference:")
#                print(Jx_arr - dF_arr)
#                print()
                
                
#                if ifield == 3 and tfield == 2:
#                    print("Jacobian:")
#                    print(Jx_arr)
#                    print()
#                    
#                    print("[F(x+dx) - F(x-dx)] / [2 eps]:")
#                    print(dF_arr)
#                    print()
                
                
                diff[tfield] = (Jx_arr - dF_arr).max()
            
            print()
        
            for tfield in range(0,nfield):
                print("max(difference)[fields=%i,%i] = %16.8E" % ( ifield, tfield, diff[tfield] ))
            
            print()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1D(args.runfile)
    petscvp.run()
#    petscvp.check_jacobian()
    
