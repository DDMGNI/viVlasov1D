'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time

from vlasov.vi.PETScMatrixFreeSimpleFunction import PETScFunction
from vlasov.vi.PETScMatrixFreeSimpleJacobian import PETScJacobian

#from vlasov.vi.PETScMatrixFreeSimpleNLFunction import PETScFunction
#from vlasov.vi.PETScMatrixFreeSimpleNLJacobian import PETScJacobian

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
        
        # create residual vector
        self.dx = self.da2.createGlobalVec()
        
        
        # create Matrix object
        self.petsc_function = PETScFunction(self.da1, self.da2, self.dax, self.h0, 
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.poisson)
        
        self.petsc_jacobian = PETScJacobian(self.da1, self.da2, self.dax, self.h0, 
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.poisson)
        
        self.J = PETSc.Mat().createPython([self.dx.getSizes(), self.b.getSizes()], comm=PETSc.COMM_WORLD)
        self.J.setPythonContext(self.petsc_jacobian)
        self.J.setUp()
        
        
        # create linear solver and preconditioner
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(self.J)
        self.ksp.setFromOptions()
        self.ksp.setType('gmres')
        self.ksp.getPC().setType('none')
#        self.ksp.setInitialGuessNonzero(True)
        
        
        # update solution history
        self.petsc_jacobian.update_history(self.x)
        self.petsc_function.update_history(self.x)
        self.vlasov_mat.update_history(self.f, self.h1)
        
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\ni = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            
            # calculate initial guess for distribution function
            self.initial_guess()
            
            # calculate function and norm
            self.petsc_function.matrix_mult(self.x, self.b)
            norm0 = self.b.norm()
            
#            for iter in range(0, self.max_iter):
#                # check if norm is smaller than tolerance
#                if norm < self.tolerance:
#                    break
                
            # update previous iteration
            self.petsc_jacobian.update_previous(self.x)
            
            # RHS = - function
            self.b.scale(-1.)
            
            # solve
            self.dx.set(0.)
            self.ksp.solve(self.b, self.dx)
            
            # add to solution vector
            self.x.axpy(1., self.dx)
            
            # calculate function and norm
            self.petsc_function.matrix_mult(self.x, self.b)
            norm1 = self.b.norm()
            
            # some solver output
            if PETSc.COMM_WORLD.getRank() == 0:
                print("     iter = %3i,  Linear Solver:  %5i iterations,   residual = %24.16E " % (0, self.ksp.getIterationNumber(), self.ksp.getResidualNorm()) )
#                print("     iter = %3i,  Linear Solver:  %5i iterations,   residual = %24.16E " % (iter, self.ksp.getIterationNumber(), self.ksp.getResidualNorm()) )
                print("                                         Initial Function Norm = %24.16E" % (norm0) )
                print("                                         Final   Function Norm = %24.16E" % (norm1) )
                
            
            # update data vectors
            self.copy_x_to_f()
            self.copy_x_to_p()
            self.copy_p_to_h()
            
            # update history
            self.petsc_function.update_history(self.x)
            self.petsc_jacobian.update_history(self.x)
            self.vlasov_mat.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
            # some solver output
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
#                print("   Nonlin Solver:       %5i iterations,   tolerance = %E " % (1, self.tolerance) )
#                print("   Nonlin Solver:  %5i iterations,   tolerance = %E " % (iter, self.tolerance) )
#                print("        sum(phi):       %24.16E" % (phisum))
                print
            
        
    
#    def initial_guess(self):
#        # calculate initial guess for distribution function
#        self.arakawa_rk4.rk4(self.f, self.h1)
#        self.copy_f_to_x()
#        
#        if PETSc.COMM_WORLD.getRank() == 0:
#            print("     RK4")
#        
#        # calculate initial guess for potential
#        self.calculate_potential()
#        
#        # correct initial guess for distribution function
#        self.calculate_vlasov()
            
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1D(args.runfile)
    petscvp.run()
    
