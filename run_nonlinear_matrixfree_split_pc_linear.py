'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse
import sys, time
import numpy as np

from petsc4py import PETSc
from run_base import petscVP1Dbase

# from vlasov.vi.PETScNLArakawaJ1            import PETScSolver
# from vlasov.vi.PETScNLArakawaJ2            import PETScSolver
from vlasov.vi.PETScNLArakawaJ4            import PETScSolver

# from vlasov.predictor.PETScPoissonMatrixJ1     import PETScPoissonMatrix
# from vlasov.predictor.PETScPoissonMatrixJ2     import PETScPoissonMatrix
from vlasov.predictor.PETScPoissonMatrixJ4     import PETScPoissonMatrix

# from vlasov.predictor.PETScNLVlasovArakawaJ1     import PETScVlasovSolver
# from vlasov.predictor.PETScNLVlasovArakawaJ2     import PETScVlasovSolver
from vlasov.predictor.PETScNLVlasovArakawaJ4     import PETScVlasovSolver

import vlasov.predictor.PETScVlasovArakawaJ4

from vlasov.predictor.PETScPoissonSolver         import PETScPoissonSolver



class petscVP1Dmatrixfree(petscVP1Dbase):
    '''
    PETSc/Python Vlasov Poisson GMRES Solver in 1D.
    '''


    def updateVlasovJacobian(self, snes, X, J, P):
        self.petsc_vlasov_solver.update_delta(X)
        
        if J != P:
            self.petsc_vlasov_solver.formJacobian(P)
        
    
    def updateLinearJacobian(self, snes, X, J, P):
        self.petsc_linear_solver.update_delta(X)
        
        if J != P:
            self.petsc_linear_solver.formJacobian(P)
        
    
    def updateJacobian(self, snes, X, J, P):
        self.petsc_solver.update_previous(X)
        
        self.petsc_solver.formJacobian(J)
        J.setNullSpace(self.nullspace)
        
        if J != P:
            self.petsc_solver.formJacobian(P)
            P.setNullSpace(self.nullspace)
        
    
    def run(self):

        OptDB = PETSc.Options()
        
        OptDB.setValue('snes_ls', 'basic')

        OptDB.setValue('ksp_monitor',  '')
#         OptDB.setValue('snes_monitor', '')
        
#        OptDB.setValue('log_info',    '')
#        OptDB.setValue('log_summary', '')
        
        # create RHS vector for predictor
        self.bpred = self.da1.createGlobalVec()
        
        # initialise predictor Jacobian
#         self.Jpc = self.da1.createMat()
#         self.Jpc.setOption(self.Jpc.Option.NEW_NONZERO_ALLOCATION_ERR, False)
#         self.Jpc.setUp()


        # create solver objects
        self.petsc_solver = PETScSolver(self.da1, self.da2, self.dax,
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.charge, coll_freq=self.coll_freq)
        
        self.petsc_vlasov_solver = PETScVlasovSolver(self.da1, self.da2, self.dax,
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.charge, coll_freq=self.coll_freq)
        
        self.petsc_linear_solver = vlasov.predictor.PETScVlasovArakawaJ4.PETScVlasovSolver(
                                            self.da1, self.da2, self.dax,
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.charge, coll_freq=self.coll_freq)
        
        
        
        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.f.getSizes(), self.bpred.getSizes()], 
                                            context=self.petsc_vlasov_solver,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
        
        
        self.Jmf_linear = PETSc.Mat().createPython([self.f.getSizes(), self.bpred.getSizes()], 
                                                   context=self.petsc_linear_solver,
                                                   comm=PETSc.COMM_WORLD)
        self.Jmf_linear.setUp()
        
        

        # copy external potential
        self.petsc_vlasov_solver.update_external(self.p_ext)
        self.petsc_linear_solver.update_external(self.p_ext)
        self.petsc_solver.update_external(self.p_ext)
        
        # update solution history
        self.petsc_vlasov_solver.update_history(self.x)
        self.petsc_linear_solver.update_history(self.x)
        self.petsc_solver.update_history(self.x)


        # create nonlinear predictor solver
        OptDB.setValue('ksp_rtol', 1E-13)

        self.snes = PETSc.SNES().create()
        self.snes.setType('ksponly')
        self.snes.setFunction(self.petsc_vlasov_solver.function_snes_mult, self.bpred)
        self.snes.setJacobian(self.updateVlasovJacobian, self.Jmf)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('gmres')
        self.snes.getKSP().getPC().setType('none')
        
        
        self.snes_pred = PETSc.SNES().create()
        self.snes_pred.setType('ksponly')
        self.snes_pred.setFunction(self.petsc_linear_solver.function_snes_mult, self.bpred)
        self.snes_pred.setJacobian(self.updateLinearJacobian, self.Jmf_linear)
        self.snes_pred.setFromOptions()
        self.snes_pred.getKSP().setType('gmres')
        self.snes_pred.getKSP().getPC().setType('none')
        
        OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        
        
#         self.petsc_poisson_solver = PETScPoissonSolver(self.da1, self.dax,
#                                             self.nx, self.nv, self.hx, self.hv,
#                                             self.charge)
#           
#         self.poisson_mf = PETSc.Mat().createPython([self.p.getSizes(), self.pb.getSizes()], 
#                                             context=self.petsc_poisson_solver,
#                                             comm=PETSc.COMM_WORLD)
#         self.poisson_mf.setUp()
#           
#           
#         del self.poisson_ksp
#            
#         OptDB.setValue('ksp_rtol', 1E-13)
#            
#         self.poisson_ksp = PETSc.KSP().create()
#         self.poisson_ksp.setFromOptions()
#         self.poisson_ksp.setOperators(self.poisson_mf, self.poisson_A)
# #         self.poisson_ksp.setType('gmres')
#         self.poisson_ksp.setType('cg')
# #         self.poisson_ksp.setType('bcgs')
# #         self.poisson_ksp.setType('ibcgs')
#         self.poisson_ksp.getPC().setType('hypre')
# #         self.poisson_ksp.getPC().setType('none')
#            
#         OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        
        
        
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # calculate external field and copy to matrix, jacobian and function
            self.calculate_external(current_time)
            self.petsc_solver.update_external(self.p_ext)
            self.petsc_vlasov_solver.update_external(self.p_ext)
            self.petsc_linear_solver.update_external(self.p_ext)
            
            
            # backup previous step
            self.x.copy(self.xh)
            
            # compute norm of previous step
            self.petsc_solver.function_mult(self.x, self.b)
            prev_norm = self.b.norm()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Previous Step:                             funcnorm = %24.16E" % (prev_norm))
            
            # calculate initial guess via RK4
            self.initial_guess_rk4()
            
            # calculate initial guess via Gear
#             self.initial_guess_gear(itime)
            
            # calculate initial guess via symplectic method
#             self.initial_guess_symplectic2()
#             self.initial_guess_symplectic4()
            
            # check if residual went down
            self.petsc_solver.function_mult(self.x, self.b)
            ig_norm = self.b.norm()
            
            # if residual of previous step is smaller then initial guess
            # copy back previous step
            if ig_norm > prev_norm:
                self.xh.copy(self.x)
            
            
#             self.petsc_vlasov_solver.update_previous(self.x)
#             self.snes.solve(None, self.f)
#             self.calculate_moments(potential=False)
#             self.calculate_potential(output=False)
#             self.copy_f_to_x()
#             self.copy_p_to_x()
#             self.copy_p_to_h()
#             
#             
#             # compute norm of predictor
#             self.petsc_solver.function_mult(self.x, self.b)
#             pred_norm = self.b.norm()
#             phisum = self.p.sum()
#             
#             if PETSc.COMM_WORLD.getRank() == 0:
#                 print("  Predictor:                          funcnorm = %24.16E" % (pred_norm))
#                 print("                                      sum(phi) = %24.16E" % (phisum))
            
            
            self.petsc_linear_solver.update_previous(self.x)
            self.snes_pred.solve(None, self.f)
            self.copy_f_to_x()
            
            self.calculate_moments(output=False)
            
            self.petsc_solver.function_mult(self.x, self.b)
            pred_norm = self.b.norm()
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Linear Solver:    %5i GMRES  iterations, funcnorm = %24.16E" % (self.snes_pred.getLinearSolveIterations(), pred_norm) )
                print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.poisson_ksp.getIterationNumber(), phisum))
                
            
            i = 0
            pred_norm = np.min([prev_norm, ig_norm])
            while True:
                i+=1
                
                self.x.copy(self.xh)
                
                self.petsc_vlasov_solver.update_previous(self.x)
                self.snes.solve(None, self.f)
                self.copy_f_to_x()
                
                self.calculate_moments(output=False)
                
                self.petsc_solver.function_mult(self.x, self.b)
                prev_norm = pred_norm
                pred_norm = self.b.norm()
                phisum = self.p.sum()

                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlinear Solver: %5i GMRES  iterations, funcnorm = %24.16E" % (self.snes.getLinearSolveIterations(), pred_norm) )
                    print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.poisson_ksp.getIterationNumber(), phisum))
                
                if pred_norm > prev_norm or pred_norm < self.cfg['solver']['petsc_snes_atol'] or i >= self.cfg['solver']['petsc_snes_max_iter']:
                    if pred_norm > prev_norm:
                        self.xh.copy(self.x)
                    
                    break
            
            # nonlinear solve
#             self.snes.solve(None, self.x)
            
            # update data vectors
            self.copy_x_to_data()
            
            # output some solver info
#             phisum = self.p.sum()
#             
#             if PETSc.COMM_WORLD.getRank() == 0:
#                 print("  Nonlinear Solver: %5i Newton iterations, funcnorm = %24.16E" % (self.snes.getIterationNumber(), self.snes.getFunctionNorm()) )
#                 print("                    %5i GMRES  iterations, sum(phi) = %24.16E" % (self.snes.getLinearSolveIterations(), phisum))
#                 print()
            
#             if self.snes.getConvergedReason() < 0:
#                 if PETSc.COMM_WORLD.getRank() == 0:
#                     print()
#                     print("Solver not converging...   %i" % (self.snes.getConvergedReason()))
#                     print()
            
            
            # update history
            self.petsc_solver.update_history(self.x)
            self.petsc_vlasov_solver.update_history(self.x)
            self.petsc_linear_solver.update_history(self.x)
            self.arakawa_gear.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
            
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1Dmatrixfree(args.runfile)
    petscvp.run()
    
