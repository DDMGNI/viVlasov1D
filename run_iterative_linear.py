'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time

from vlasov.vi.PETScMatrixFreeSimple import PETScSolver

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
        self.petsc_mat = PETScSolver(self.da1, self.da2, self.dax, self.day,
                                     self.x, self.b, self.h0, self.vGrid,
                                     self.nx, self.nv, self.ht, self.hx, self.hv,
                                     self.poisson, self.alpha)
        
        self.A = PETSc.Mat().createPython([self.x.getSizes(), self.b.getSizes()], comm=PETSc.COMM_WORLD)
        self.A.setPythonContext(self.petsc_mat)
        self.A.setUp()

        # create linear solver and preconditioner
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.A)
        self.ksp.setType('gmres')
        self.ksp.getPC().setType('none')
        self.ksp.setInitialGuessNonzero(True)
        
        
        # update solution history
        self.petsc_mat.update_history(self.f, self.h1)
        self.vlasov_mat.update_history(self.f, self.h1)
        
        
        
    
    def run(self):
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            # build RHS
            self.petsc_mat.formRHS(self.b)
            
            # calculate initial guess for distribution function
            self.initial_guess()
            
            # solve
            self.ksp.solve(self.b, self.x)
            
            # update data vectors
            self.copy_x_to_f()
            self.copy_x_to_p()
            self.copy_p_to_h()
            
            # update history
            self.petsc_mat.update_history(self.f, self.h1)
            self.vlasov_mat.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
            
            # some solver output
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("     Solver:   %5i iterations,   residual = %24.16E " % (self.ksp.getIterationNumber(), self.ksp.getResidualNorm()) )
                print("                                   sum(phi) = %24.16E" % (phisum))
                print
                
#            if self.ksp.getIterationNumber() == self.max_iter:
#                break
            
#    def initial_guess(self):
#        self.arakawa_rk4.rk4(self.f, self.h1)
#        self.copy_f_to_x()
#        
#        if PETSc.COMM_WORLD.getRank() == 0:
#            print("     RK4")
#
#        # calculate initial guess for potential
#        self.calculate_potential()
#        self.vlasov_mat.update_potential(self.h1)

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1D(args.runfile)
    petscvp.run()
    