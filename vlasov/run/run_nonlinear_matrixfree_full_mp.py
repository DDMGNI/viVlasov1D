'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse
import sys, time
import numpy as np

from petsc4py import PETSc

from vlasov.run.run_base_full import viVlasov1Dbasefull

# from vlasov.solvers.full.PETScNLArakawaJ1            import PETScSolver
# from vlasov.solvers.full.PETScNLArakawaJ2            import PETScSolver
from vlasov.solvers.full.PETScNLArakawaJ4            import PETScSolver


# solver_package = 'superlu_dist'
solver_package = 'mumps'
# solver_package = 'pastix'


class viVlasov1Drunscript(viVlasov1Dbasefull):
    '''
    PETSc/Python Vlasov Poisson GMRES Solver in 1D.
    '''

    def updateMatrix(self, snes, X, J, P):
        self.petsc_matrix.formMat(J)
        J.setNullSpace(self.nullspace)
        
        if J != P:
            self.petsc_matrix.formMat(P)
            P.setNullSpace(self.nullspace)
    
    
    def updateJacobian(self, snes, X, J, P):
        self.petsc_solver.update_previous(X)
        
        if J != P:
            self.petsc_solver.formJacobian(P)
            P.setNullSpace(self.nullspace)
        
    
    def initial_guess(self):
        self.snes_linear.solve(None, self.x)
        self.copy_x_to_data()
        
        self.calculate_moments(potential=False)
        
        self.petsc_function.mult(self.x, self.b)
        ignorm = self.b.norm()
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Linear Solver:                      funcnorm = %24.16E" % (ignorm))
            print("                                      sum(phi) = %24.16E" % (phisum))
        
    
    def run(self):
        
        OptDB = PETSc.Options()
        
#         OptDB.setValue('ksp_monitor',  '')
#         OptDB.setValue('snes_monitor', '')
        
#         OptDB.setValue('snes_lag_preconditioner', 3)
        
#         OptDB.setValue('snes_ls', 'basic')


        # create Jacobian, Function, and linear Matrix objects
        self.petsc_solver = PETScSolver(self.da1, self.da2, self.dax,
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.charge, coll_freq=self.coll_freq)
        

        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.x.getSizes(), self.b.getSizes()], 
                                            context=self.petsc_solver,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
#         self.Jmf.setNullSpace(self.nullspace)
        
        

        # copy external potential
        self.petsc_solver.update_external(self.p_ext)
        
        # update solution history
        self.petsc_solver.update_history(self.x)


        # create nonlinear solver
        self.snes = PETSc.SNES().create()
        self.snes.setFunction(self.petsc_solver.function_snes_mult, self.b)
        self.snes.setJacobian(self.updateJacobian, self.Jmf)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('gmres')
        self.snes.getKSP().getPC().setType('none')
#         self.snes.getKSP().getPC().setType('bjacobi')
#         self.snes.getKSP().getPC().setType('asm')
#         self.snes.getKSP().setType('preonly')
#         self.snes.getKSP().getPC().setType('lu')
#         self.snes.getKSP().getPC().setFactorSolverPackage(solver_package)
        
#        self.snes_nsp = PETSc.NullSpace().create(vectors=(self.x_nvec,))
#        self.snes.getKSP().setNullSpace(self.snes_nsp)
        
        
        
        
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # calculate external field and copy to matrix, jacobian and function
            self.calculate_external(current_time)
#            self.petsc_jacobian_mf.update_external(self.p_ext)
            self.petsc_solver.update_external(self.p_ext)
#             self.petsc_jacobian.update_external(self.p_ext)
#             self.petsc_function.update_external(self.p_ext)
#             self.petsc_matrix.update_external(self.p_ext)
            
            
            self.x.copy(self.xh)
            
            self.petsc_solver.function_mult(self.x, self.b)
            prev_norm = self.b.norm()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Previous Step:                          funcnorm = %24.16E" % (prev_norm))
            
            # calculate initial guess via RK4
#             self.initial_guess_rk4()
            
            # calculate initial guess via Gear
            self.initial_guess_gear(itime)
            
            # check if residual went down
#             self.petsc_function.mult(self.x, self.b)
#             ig_norm = self.b.norm()
#             
#             if ig_norm > prev_norm:
#                 self.xh.copy(self.x)
            
            
            # calculate initial guess via linear solver
#            self.initial_guess()
            
            
            # nonlinear solve
            self.snes.solve(None, self.x)
            
            # output some solver info
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Nonlinear Solver: %5i Newton iterations, funcnorm = %24.16E" % (self.snes.getIterationNumber(), self.snes.getFunctionNorm()) )
                print("                    %5i GMRES  iterations, sum(phi) = %24.16E" % (self.snes.getLinearSolveIterations(), phisum))
                print()
            
            if self.snes.getConvergedReason() < 0:
                if PETSc.COMM_WORLD.getRank() == 0:
                    print()
                    print("Solver not converging...   %i" % (self.snes.getConvergedReason()))
                    print()
            
            
#             if PETSc.COMM_WORLD.getRank() == 0:
#                 mat_viewer = PETSc.Viewer().createDraw(size=(800,800), comm=PETSc.COMM_WORLD)
#                 mat_viewer(self.J)
#                 
#                 print
#                 input('Hit any key to continue.')
#                 print
            
            
            # update data vectors
            self.copy_x_to_data()
            
            # update history
            self.petsc_solver.update_history(self.x)
#             self.petsc_matrix.update_history(self.f, self.h1, self.p, self.n, self.nu, self.ne, self.u, self.e, self.a)
            self.arakawa_gear.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
                        
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = viVlasov1Drunscript(args.runfile)
    petscvp.run()
    
