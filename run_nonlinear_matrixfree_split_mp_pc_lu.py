'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse, time

from petsc4py import PETSc

from run_base_split import petscVP1Dbasesplit


class petscVP1Dmatrixfree(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson GMRES Solver in 1D.
    '''


    def __init__(self, cfgfile, runid=None, cfg=None):
        super().__init__(cfgfile, runid, cfg)
        
        # create solver objects
        self.vlasov_solver = self.vlasov_object.PETScVlasovSolver(
                                               self.da1, self.dax,
                                               self.h0, self.vGrid,
                                               self.nx, self.nv, self.ht, self.hx, self.hv,
                                               self.charge, coll_freq=self.coll_freq)
        
        self.vlasov_solver_kinetic = PETScVlasovSolverKinetic(self.da1, self.dax,
                                               self.h0, self.vGrid,
                                               self.nx, self.nv, self.ht, self.hx, self.hv,
                                               self.charge, coll_freq=self.coll_freq)
        
        
        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.f.getSizes(), self.fb.getSizes()], 
                                            context=self.vlasov_solver,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
        
        
        # update solution history
        self.vlasov_solver.update_history(self.f, self.p, self.p_ext, self.n, self.u, self.e)

        # initialise kinetic Jacobian
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Create and Initialise Preconditioning Matrix.")
            
        self.J = self.da1.createMat()
        self.J.setUp()

        self.vlasov_solver_kinetic.formJacobian(self.J)
        

        # create nonlinear solver
        self.snes = PETSc.SNES().create()
        self.snes.setType('ksponly')
        self.snes.setFunction(self.vlasov_solver.function_snes_mult, self.fb)
        self.snes.setJacobian(self.updateVlasovJacobian, self.Jmf, self.J)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('gmres')
        self.snes.getKSP().getPC().setType('lu')
        self.snes.getKSP().getPC().setFactorSolverPackage(self.solver_package)
        
        
        del self.poisson_ksp
        del self.poisson_solver
            
        self.poisson_solver = self.poisson_object.PETScPoissonSolver(self.dax, self.nx, self.hx, self.charge)
        self.poisson_solver.formMat(self.poisson_matrix)
        
        self.poisson_mf = PETSc.Mat().createPython([self.p.getSizes(), self.pb.getSizes()], 
                                                   context=self.poisson_solver,
                                                   comm=PETSc.COMM_WORLD)
        self.poisson_mf.setUp()
           
           
        OptDB.setValue('ksp_rtol', 1E-13)
            
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_mf, self.poisson_matrix)
        self.poisson_ksp.setType('cg')
#         self.poisson_ksp.setType('bcgs')
#         self.poisson_ksp.setType('ibcgs')
        self.poisson_ksp.getPC().setType('hypre')
            
        OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        
        
    
    def updateVlasovJacobian(self, snes, X, J, P):
        return PETSc.Mat().Structure.SAME_PRECONDITIONER
    
    
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
                
#                 if i == 1:
#                     self.snes.getKSP().setTolerances(rtol=1E-5)
#                 if i == 2:
#                     self.snes.getKSP().setTolerances(rtol=1E-4)
#                 if i == 3:
#                     self.snes.getKSP().setTolerances(rtol=1E-3)
#                 if i == 4:
#                     self.snes.getKSP().setTolerances(rtol=1E-3)
                
                
                self.f.copy(self.fh)
                
                self.vlasov_solver.update_previous(self.f, self.p, self.p_ext, self.n, self.u, self.e)
                self.snes.solve(None, self.f)
                
                self.calculate_moments(output=False)
                self.vlasov_solver.update_previous(self.f, self.p, self.p_ext, self.n, self.u, self.e)
                
                prev_norm = pred_norm
                pred_norm = self.calculate_residual()
                phisum = self.p.sum()

                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlinear Solver: %5i GMRES  iterations, funcnorm = %24.16E" % (self.snes.getLinearSolveIterations(), pred_norm) )
                    print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.poisson_ksp.getIterationNumber(), phisum))
                
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
    
