'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse, time

from petsc4py import PETSc

# from vlasov.solver.vlasov.PETScNLVlasovArakawaJ1          import PETScVlasovSolver
# from vlasov.solver.vlasov.PETScNLVlasovArakawaJ2          import PETScVlasovSolver
from vlasov.solver.vlasov.PETScNLVlasovArakawaJ4          import PETScVlasovSolver
from  vlasov.solver.vlasov.PETScNLVlasovArakawaJ4kinetic  import PETScVlasovSolverKinetic

from run_base_split import petscVP1Dbasesplit


# solver_package = 'superlu_dist'
solver_package = 'mumps'
# solver_package = 'pastix'

class petscVP1Dmatrixfree(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson GMRES Solver in 1D.
    '''


    def __init__(self, cfgfile, runid):
        super().__init__(cfgfile, runid)
        
        OptDB = PETSc.Options()
        
#         OptDB.setValue('snes_ls', 'basic')

#         OptDB.setValue('ksp_monitor',  '')
#         OptDB.setValue('snes_monitor', '')
        
#         OptDB.setValue('log_info',    '')
#         OptDB.setValue('log_summary', '')
        
        
        # create some more vectors
        self.tb = self.da1.createGlobalVec()
        self.tx = self.da1.createGlobalVec()
        self.ty = self.da1.createGlobalVec()
        self.tz = self.da1.createGlobalVec()
        
        
        # create solver objects
        self.vlasov_solver = PETScVlasovSolver(self.da1, self.dax,
                                               self.h0, self.vGrid,
                                               self.nx, self.nv, self.ht, self.hx, self.hv,
                                               self.charge, coll_freq=self.coll_freq)
        
        self.vlasov_solver_kinetic = PETScVlasovSolverKinetic(self.da1, self.dax,
                                               self.h0, self.vGrid,
                                               self.nx, self.nv, self.ht, self.hx, self.hv,
                                               self.charge, coll_freq=self.coll_freq)
        
        
        # initialise matrix-free Jacobian
        self.Jmf = PETSc.Mat().createPython([self.f.getSizes(), self.fb.getSizes()], 
                                            context=self,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
        
        
        self.Jmf_kinetic = PETSc.Mat().createPython([self.f.getSizes(), self.fb.getSizes()], 
                                                    context=self.vlasov_solver_kinetic,
                                                    comm=PETSc.COMM_WORLD)
        self.Jmf_kinetic.setUp()
        
        
        # update solution history
        self.vlasov_solver.update_history(self.f, self.p, self.p_ext, self.n, self.u, self.e)


        # initialise kinetic Jacobian
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Create and Initialise Preconditioning Matrix.")
            
        self.J = self.da1.createMat()
        self.J.setUp()

        self.vlasov_solver_kinetic.formJacobian(self.J)
        
        
        # initial guess with kinetic Jacobian
        self.ksp_linear = PETSc.KSP().create()
        self.ksp_linear.setFromOptions()
        self.ksp_linear.setOperators(self.J)
#         self.ksp_linear.setType('gmres')
#         self.ksp_linear.getPC().setType('none')
        self.ksp_linear.setType('preonly')
        self.ksp_linear.getPC().setType('lu')
        self.ksp_linear.getPC().setFactorSolverPackage(solver_package)
        
        
        # create matrix-free Poisson solver
        self.poisson_mf = PETSc.Mat().createPython([self.p.getSizes(), self.pb.getSizes()], 
                                                   context=self.poisson_solver,
                                                   comm=PETSc.COMM_WORLD)
        self.poisson_mf.setUp()
           
           
        del self.poisson_ksp
            
        OptDB.setValue('ksp_rtol', 1E-13)
            
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_mf, self.poisson_A)
        self.poisson_ksp.setType('cg')
#         self.poisson_ksp.setType('bcgs')
#         self.poisson_ksp.setType('ibcgs')
        self.poisson_ksp.getPC().setType('hypre')
            
        OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        
        
    
    def updateVlasovJacobian(self, snes, X, J, P):
        return PETSc.Mat().Structure.SAME_PRECONDITIONER
    
    
    def mult(self, mat, X, Y):
        self.vlasov_solver.jacobian_mult(X, self.ty)
        self.ksp_linear.solve(self.ty, Y)
        
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # calculate external field and copy to solver
            self.calculate_external(current_time)
            self.vlasov_solver.update_previous(self.f, self.p, self.p_ext, self.n, self.u, self.e)
            
            # compute initial guess
            self.initial_guess()
            
            # nonlinear solve
            i = 0
            pred_norm = self.calculate_residual()
            while True:
                i+=1
                
                # backup
                self.f.copy(self.fh)
                
                # update previous (i.e., explicit initial guess)
                self.vlasov_solver.update_previous(self.f, self.p, self.p_ext, self.n, self.u, self.e)
            
                # compute right hand side
                self.vlasov_solver.function_mult(self.f, self.fb)
                self.fb.scale(-1)
                
                # precondition right hand side
                self.ksp_linear.solve(self.fb, self.tb)
                
                # solve
                ksp = PETSc.KSP().create()
                ksp.setFromOptions()
                ksp.setOperators(self.Jmf)
                ksp.setType('gmres')
                ksp.getPC().setType('none')
#                 ksp.setInitialGuessNonzero(True)
                
                if i == 1:
                    ksp.setTolerances(rtol=1E-5)
                if i == 2:
                    ksp.setTolerances(rtol=1E-4)
                if i == 3:
                    ksp.setTolerances(rtol=1E-3)
                if i == 4:
                    ksp.setTolerances(rtol=1E-3)
                
                ksp.solve(self.tb, self.df)
                
                self.f.axpy(1, self.df)
                
                self.calculate_moments(output=False)
                self.vlasov_solver.update_previous(self.f, self.p, self.p_ext, self.n, self.u, self.e)
                
                prev_norm = pred_norm
                pred_norm = self.calculate_residual()
                phisum = self.p.sum()

                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlinear Solver: %5i GMRES  iterations, funcnorm = %24.16E" % (ksp.getIterationNumber(), pred_norm) )
                    print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.poisson_ksp.getIterationNumber(), phisum))
                
                del ksp

                if pred_norm > prev_norm or pred_norm < self.cfg['solver']['petsc_snes_atol'] or i >= self.cfg['solver']['petsc_snes_max_iter']:
                    if pred_norm > prev_norm:
                        self.fh.copy(self.f)
                        self.calculate_moments(output=False)
                    
                    break
            
            # update history
            self.vlasov_solver.update_history(self.f, self.p, self.p_ext, self.n, self.u, self.e)
            self.arakawa_gear.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('-c', metavar='<cfg_file>', type=str,
                        help='Configuration File')
    parser.add_argument('-i', metavar='<runid>', type=str, default="",
                        help='Run ID')
    
    args = parser.parse_args()
    
    petscvp = petscVP1Dmatrixfree(args.c, args.i)
    petscvp.run()
    
