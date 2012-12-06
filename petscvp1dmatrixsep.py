'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time


from vlasov.predictor.PETScPoissonMatrix import PETScPoissonMatrix
from vlasov.predictor.PETScVlasovMatrix  import PETScMatrix

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
        
        
        # create Matrix object
        self.petsc_mat = PETScMatrix(self.da1, self.dax,
                                     self.h0, self.vGrid,
                                     self.nx, self.nv, self.ht, self.hx, self.hv,
                                     self.alpha)
        
        self.A = self.da1.createMat()
        self.A.setType('mpiaij')
        self.A.setUp()

        # create linear solver and preconditioner
#        self.ksp = PETSc.KSP().create()
#        self.ksp.setFromOptions()
#        self.ksp.setOperators(self.A)
##        self.ksp.setType('gmres')
#        self.ksp.setType('preonly')
##        self.ksp.getPC().setType('none')
#        self.ksp.getPC().setType('lu')
##        self.ksp.getPC().setFactorSolverPackage('superlu_dist')
#        self.ksp.getPC().setFactorSolverPackage('mumps')
##        self.ksp.setInitialGuessNonzero(True)
        
        
        
        self.poisson_A = self.dax.createMat()
        self.poisson_A.setType('mpiaij')
        self.poisson_A.setUp()
        
        self.poisson_mat = PETScPoissonMatrix(self.da1, self.dax, 
                                              self.nx, self.nv, self.hx, self.hv,
                                              self.poisson)
        self.poisson_mat.formMat(self.poisson_A)
        
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_A)
        self.poisson_ksp.setType('preonly')
        self.poisson_ksp.getPC().setType('lu')
#        self.poisson_ksp.getPC().setFactorSolverPackage('superlu_dist')
        self.poisson_ksp.getPC().setFactorSolverPackage('mumps')
        
        self.poisson_nsp = PETSc.NullSpace().create(constant=True)
        self.poisson_ksp.setNullSpace(self.poisson_nsp)        
        
        
        # solve for initial potential
        self.calculate_potential()
        
        # create history vectors
        self.fh  = self.da1.createGlobalVec()
        self.h1h = self.da1.createGlobalVec()
        
        
        # update solution history
        self.vlasov_mat.update_history(self.f, self.h1)
        
        
    
    def calculate_potential(self):
        self.poisson_mat.formRHS(self.f, self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        phisum = self.p.sum()
        
        self.copy_p_to_x()
        self.copy_p_to_h()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Poisson:  %5i iterations,   residual = %24.16E" % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
            print("                                   sum(phi) = %24.16E" % (phisum))
    
    
    def calculate_vlasov(self):
    
        # build matrix
        self.petsc_mat.formMat(self.A, self.h1h)
        
#        mat_viewer = PETSc.Viewer().createDraw(size=(800,800), comm=PETSc.COMM_WORLD)
#        mat_viewer(self.A)
#        
#        print
#        input('Hit any key to continue.')
#        print
        
        # build RHS
        self.petsc_mat.formRHS(self.fb, self.fh, self.h1)
        
        # create solver
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.A)
#        self.ksp.setType('gmres')
        self.ksp.setType('preonly')
#        self.ksp.getPC().setType('none')
        self.ksp.getPC().setType('lu')
        self.ksp.getPC().setFactorSolverPackage('superlu_dist')
#        self.ksp.getPC().setFactorSolverPackage('mumps')
#        self.ksp.setInitialGuessNonzero(True)
        
        # solve Vlasov equation
        self.ksp.solve(self.fb, self.f)
        
        # update data vectors
        self.copy_f_to_x()
        
        # some solver output
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Solver:   %5i iterations,   residual = %24.16E " % (self.ksp.getIterationNumber(), self.ksp.getResidualNorm()) )
    
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            # save history
            self.f.copy(self.fh)
            self.h1.copy(self.h1h)
            
            # calculate initial guess
            self.initial_guess()
            
            # solve Vlasov equation
            self.calculate_vlasov()
            
#            # build matrix
#            self.petsc_mat.formMat(self.A, self.h1h)
#            
#            if itime == 1:
#                
#                mat_viewer = PETSc.Viewer().createDraw(size=(800,800), comm=PETSc.COMM_WORLD)
#                mat_viewer(self.A)
#                
#                print
#                input('Hit any key to continue.')
#                print
            
            
#            # build RHS
#            self.petsc_mat.formRHS(self.fb, self.fh, self.h1)
#            
#            # solve Vlasov equation
#            self.ksp.solve(self.fb, self.f)
#
#            # update data vectors
#            self.copy_f_to_x()
            
#            # some solver output
#            if PETSc.COMM_WORLD.getRank() == 0:
#                print("     Solver:   %5i iterations,   residual = %24.16E " % (self.ksp.getIterationNumber(), self.ksp.getResidualNorm()) )
                
            # update potential
            self.calculate_potential()
            
#            # update distribution function
#            self.calculate_vlasov()
#            
#            # build RHS
#            self.petsc_mat.formRHS(self.fb, self.fh, self.h1)
#            
#            # solve Vlasov equation
#            self.ksp.solve(self.fb, self.f)
#
#            # update potential
#            self.calculate_potential()
            
            # update history
            self.vlasov_mat.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
        
    
    def initial_guess(self):
        self.arakawa_rk4.rk4(self.f, self.h1)
        self.copy_f_to_x()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     RK4")

        # calculate initial guess for potential
        self.calculate_potential()
        self.vlasov_mat.update_potential(self.h1)

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1D(args.runfile)
    petscvp.run()
    
