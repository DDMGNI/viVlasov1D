'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse, time

from petsc4py import PETSc

from vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4RK4    import PETScVlasovSolver
from vlasov.solvers.poisson.PETScPoissonSolver4         import PETScPoissonSolver

from run_base_split_rk4 import petscVP1DbasesplitRK4


class petscVP1Dmatrixfree(petscVP1DbasesplitRK4):
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
        
        
        # create solver objects
        self.vlasov_solver = PETScVlasovSolver(self.da2, self.da1, self.dax,
                                               self.h0, self.vGrid,
                                               self.nx, self.nv, self.ht, self.hx, self.hv,
                                               self.charge, coll_freq=self.coll_freq)
        
        
        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.k.getSizes(), self.b.getSizes()], 
                                            context=self.vlasov_solver,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
        
        

        # update solution history
        self.vlasov_solver.update_history(self.f, self.p, self.p_ext, self.n, self.u, self.e)


        # create nonlinear predictor solver
        self.snes = PETSc.SNES().create()
        self.snes.setType('ksponly')
        self.snes.setFunction(self.vlasov_solver.function_snes_mult, self.b)
        self.snes.setJacobian(self.updateVlasovJacobian, self.Jmf)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('gmres')
        self.snes.getKSP().getPC().setType('none')
        
        
        
        del self.poisson_ksp
        del self.poisson_solver
            
        self.poisson_solver = PETScPoissonSolver(self.dax, self.nx, self.hx, self.charge)
        self.poisson_solver.formMat(self.poisson_A)
        
        self.poisson_mf = PETSc.Mat().createPython([self.p.getSizes(), self.pb.getSizes()], 
                                                   context=self.poisson_solver,
                                                   comm=PETSc.COMM_WORLD)
        self.poisson_mf.setUp()
           
           
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
        if J != P:
            self.vlasov_solver.formJacobian(P)
        
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            current_time0 = self.ht*itime
            current_time1 = self.ht*(itime - 1 + self.c1)
            current_time2 = self.ht*(itime - 1 + self.c2)
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f  (%6.4f, %6.4f),   %s" % (itime, current_time0, current_time1, current_time2, localtime) )
                print
                self.time.setValue(0, current_time0)
            
            # compute initial guess
            self.initial_guess4()
            
            # calculate external field and copy to solver
            self.calculate_external(current_time0, self.p_ext )
            self.calculate_external(current_time1, self.p1_ext)
            self.calculate_external(current_time2, self.p2_ext)
            
            self.vlasov_solver.update_previous4(self.f1, self.f2, self.p1, self.p2, self.p1_ext, self.p2_ext, self.n, self.u, self.e)
            
            # nonlinear solve
            i = 0
            pred_norm = self.calculate_residual4()
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
                
                
                self.k.copy(self.kh)
                
                self.snes.solve(None, self.k)
                
                self.calculate_moments4(output=False)
                self.vlasov_solver.update_previous4(self.f1, self.f2, self.p1, self.p2, self.p1_ext, self.p2_ext, self.n, self.u, self.e)
                
                prev_norm = pred_norm
                pred_norm = self.calculate_residual4()
                phisum1 = self.p1.sum()
                phisum2 = self.p2.sum()

                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlinear Solver: %5i GMRES  iterations, residual = %24.16E" % (self.snes.getLinearSolveIterations(),  pred_norm) )
                    print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.p1_niter, phisum1))
                    print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.p2_niter, phisum2))
                
                if pred_norm > prev_norm or pred_norm < self.cfg['solver']['petsc_snes_atol'] or i >= self.cfg['solver']['petsc_snes_max_iter']:
                    if pred_norm > prev_norm:
                        self.kh.copy(self.k)
                        self.calculate_moments4(output=False)
                    
                    break
            
            # compute final distribution function, potential and moments
#             self.fh.copy(self.f)
            self.f.axpy(self.b1 * self.ht, self.k1)
            self.f.axpy(self.b2 * self.ht, self.k2)
            
            self.calculate_moments(output=False)
            phisum = self.p.sum()
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Poisson   Solver: %5i CG     iterations, sum(phi) = %24.16E" % (self.poisson_ksp.getIterationNumber(), phisum)    )
            
            # update history
            self.f.copy(self.fh)
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
    