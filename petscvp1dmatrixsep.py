'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time


from vlasov.predictor.PETScVlasovMatrix import PETScMatrix

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
#        self.A.setType('seqaij')
        self.A.setUp()

        # create linear solver and preconditioner
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.A)
#        self.ksp.setType('gmres')
        self.ksp.setType('preonly')
#        self.ksp.getPC().setType('none')
        self.ksp.getPC().setType('lu')
#        self.ksp.getPC().setFactorSolverPackage('superlu_dist')
        self.ksp.getPC().setFactorSolverPackage('mumps')
#        self.ksp.setInitialGuessNonzero(True)
        
#        self.poisson_ksp.setInitialGuessNonzero(True)
        
        # create history vectors
        self.fh  = self.da1.createGlobalVec()
        self.h1h = self.da1.createGlobalVec()
        
        
        # update solution history
        self.vlasov_mat.update_history(self.f, self.h1)
        
        
    
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
            
            # build matrix
            self.petsc_mat.formMat(self.A, self.h1h)
            
            # build RHS
            self.petsc_mat.formRHS(self.fb, self.fh, self.h1)
            
            # solve Vlasov equation
            self.ksp.solve(self.fb, self.f)

            # update data vectors
            self.copy_f_to_x()
            
            # some solver output
            if PETSc.COMM_WORLD.getRank() == 0:
                print("     Solver:   %5i iterations,   residual = %24.16E " % (self.ksp.getIterationNumber(), self.ksp.getResidualNorm()) )
                
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
    
