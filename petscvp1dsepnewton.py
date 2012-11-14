'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time

from vlasov.predictor.PETScVlasovFunction import PETScVlasovFunction
from vlasov.predictor.PETScVlasovJacobian import PETScVlasovJacobian

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
        
        # create residual vector for Vlasov solver
        self.df = self.da1.createGlobalVec()
        
        
        # create Matrix objects
        self.vlasov_function = PETScVlasovFunction(self.da1, self.dax, self.h0, 
                                                   self.nx, self.nv, self.ht, self.hx, self.hv)
        
        self.vlasov_jacobian = PETScVlasovJacobian(self.da1, self.dax, self.h0, 
                                                   self.nx, self.nv, self.ht, self.hx, self.hv)
        
        
        self.J = PETSc.Mat().createPython([self.f.getSizes(), self.fb.getSizes()], comm=PETSc.COMM_WORLD)
        self.J.setPythonContext(self.vlasov_jacobian)
        self.J.setUp()
        
        
        # update solution history
        self.vlasov_jacobian.update_history(self.f, self.h1)
        self.vlasov_function.update_history(self.f, self.h1)
        
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\ni = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            
            # calculate initial guess for distribution function
            self.initial_guess()
            
#            for iter in range(0, self.max_iter):
#                # check if norm is smaller than tolerance
#                if norm < self.tolerance:
#                    break
            
            
            print
            print("     iter = %3i" % (1)) 
#            print("     iter = %3i" % (iter)) 
            
            # update previous iteration
            self.vlasov_jacobian.update_previous(self.f, self.h1)
        
            # calculate function and norm
            self.vlasov_function.matrix_mult(self.f, self.h1, self.fb)
            norm0 = self.fb.norm()
            
            # RHS = - function
            self.fb.scale(-1.)
            
            # solve
            self.df.set(0.)
            self.vlasov_ksp.solve(self.fb, self.df)
            
            # add to solution vector
            self.f.axpy(1., self.df)
            
            # calculate function and norm
            self.vlasov_function.matrix_mult(self.f, self.h1, self.fb)
            norm1 = self.fb.norm()
            
            # some solver output
            if PETSc.COMM_WORLD.getRank() == 0:
                print("     Vlasov Solver:   %5i iterations,   residual = %24.16E " % (self.vlasov_ksp.getIterationNumber(), self.vlasov_ksp.getResidualNorm()) )
                print("                                         Initial Function Norm = %24.16E" % (norm0) )
                print("                                         Final   Function Norm = %24.16E" % (norm1) )
                
            # update potential
            self.calculate_potential()
            
            
            # update history
            self.vlasov_function.update_history(self.f, self.h1)
            self.vlasov_jacobian.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
#            # some solver output
#            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
#                print("   Nonlin Solver:       %5i iterations,   tolerance = %E " % (1, self.tolerance) )
#                print("   Nonlin Solver:  %5i iterations,   tolerance = %E " % (iter, self.tolerance) )
#                print("        sum(phi):       %24.16E" % (phisum))
                print
            
        
    
#    def initial_guess(self):
#        # calculate initial guess for distribution function
#        self.arakawa_rk4.rk4(self.f, self.h1)
#        
#        if PETSc.COMM_WORLD.getRank() == 0:
#            print("     RK4 predictor")
#        
#        self.calculate_potential()
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1D(args.runfile)
    petscvp.run()
    
