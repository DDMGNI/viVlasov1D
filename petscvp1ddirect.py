'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time


from vlasov.predictor.PETScPoissonMatrix  import PETScPoissonMatrix

from vlasov.vi.PETScMatrixSimple          import PETScMatrix

from petscvp1d import petscVP1Dbase


class petscVP1D(petscVP1Dbase):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        self.eps = 0.0
#        self.eps = 1.0E-3
        
        
        # initialise parent object
        petscVP1Dbase.__init__(self, cfgfile)
        
        
        # create Matrix object
        self.petsc_mat = PETScMatrix(self.da1, self.da2, self.dax, self.day,
                                     self.h0, self.vGrid,
                                     self.nx, self.nv, self.ht, self.hx, self.hv,
                                     self.poisson, self.eps, self.alpha)
        
        self.A = self.da2.createMat()
        self.A.setType('mpiaij')
        self.A.setUp()
        
        self.A.setOption(self.A.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        

        # create linear solver and preconditioner
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.A)
        self.ksp.setType('preonly')
        self.ksp.getPC().setType('lu')
#        self.ksp.getPC().setFactorSolverPackage('superlu_dist')
        self.ksp.getPC().setFactorSolverPackage('mumps')
        
        
        
        self.poisson_mat = PETScPoissonMatrix(self.da1, self.dax, 
                                              self.nx, self.nv, self.hx, self.hv,
                                              self.poisson, self.eps)
        
        self.poisson_A = self.dax.createMat()
        self.poisson_A.setType('mpiaij')
        self.poisson_A.setUp()
        
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_A)
        self.poisson_ksp.setType('preonly')
        self.poisson_ksp.getPC().setType('lu')
#        self.poisson_ksp.getPC().setFactorSolverPackage('superlu_dist')
        self.poisson_ksp.getPC().setFactorSolverPackage('mumps')
        
        
#        # vectors for residual calculation
#        self.y  = self.da2.createGlobalVec()
#        self.fy = self.da1.createGlobalVec()
#        self.py = self.dax.createGlobalVec()
        
        # calculate initial potential
        self.calculate_potential()
        
        # update solution history
        self.petsc_mat.update_history(self.f, self.h1)
        self.vlasov_mat.update_history(self.f, self.h1)
        
        # save to hdf5
        self.hdf5_viewer.HDF5SetTimestep(0)
        self.save_hdf5_vectors()
        
        
        
    def calculate_potential(self):
        
        self.poisson_mat.formMat(self.poisson_A)
        self.poisson_mat.formRHS(self.f, self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        phisum = self.p.sum()
        
        self.remove_average_from_potential()
        
        self.copy_p_to_x()
        self.copy_p_to_h()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Poisson:  %5i iterations,   residual = %24.16E" % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
            print("                                   sum(phi) = %24.16E" % (phisum))
    
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            self.ksp = PETSc.KSP().create()
            self.ksp.setFromOptions()
            self.ksp.setOperators(self.A)
            self.ksp.setType('preonly')
            self.ksp.getPC().setType('lu')
            self.ksp.getPC().setFactorSolverPackage('mumps')
            
            
            # build matrix
            self.petsc_mat.formMat(self.A)
            
            # build RHS
            self.petsc_mat.formRHS(self.b)
            
#            if itime == 1:
##                self.A.view()
##                
##                self.b.view()
#                
#                mat_viewer = PETSc.Viewer().createDraw(size=(800,800), comm=PETSc.COMM_WORLD)
#                mat_viewer(self.A)
#                
#                print
#                raw_input('Hit any key to continue.')
#                print
            
            # solve
            self.ksp.solve(self.b, self.x)
            
            # update data vectors
            self.copy_x_to_f()
            self.copy_x_to_p()
            
            self.remove_average_from_potential()
        
            self.copy_p_to_x()
            self.copy_p_to_h()
            
            
#            # calculate residuals
#            self.A.mult(self.x, self.y)
#            self.y.axpy(-1., self.b)
#            res_solver = self.y.norm()
#            
#            self.vlasov_mat.update_potential(self.h1)
#            self.vlasov_mat.matrix_mult(self.f, self.fy)
#            self.vlasov_mat.formRHS(self.fb)
#            self.fy.axpy(-1., self.fb)
#            res_vlasov = self.fy.norm()
#            
#            self.poisson_A.mult(self.p, self.py)
#            self.poisson_mat.formRHS(self.f, self.pb)
#            self.py.axpy(-1., self.pb)
#            res_poisson = self.py.norm()
            
            
            # update history
            self.petsc_mat.update_history(self.f, self.h1)
            self.vlasov_mat.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
            
#            # some solver output
            phisum = self.p.sum()
            
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("     Solver")
                print("     sum(phi) = %24.16E" % (phisum))
#                print("     Solver:   %5i iterations,   sum(phi) = %24.16E" % (phisum))
#                print("                               res(solver)  = %24.16E" % (res_solver))
#                print("                               res(vlasov)  = %24.16E" % (res_vlasov))
#                print("                               res(poisson) = %24.16E" % (res_poisson))
#                print
                
            self.ksp.destroy()
            
#            if self.ksp.getIterationNumber() == self.max_iter:
#                break
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1D(args.runfile)
    petscvp.run()
    
