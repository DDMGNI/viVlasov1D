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
        
#         OptDB = PETSc.Options()
        
#         OptDB.setValue('snes_ls', 'basic')

#         OptDB.setValue('ksp_monitor',  '')
#         OptDB.setValue('snes_monitor', '')
        
#         OptDB.setValue('log_info',    '')
#         OptDB.setValue('log_summary', '')
        
        
        # create solver objects
        self.vlasov_solver = PETScVlasovSolver(self.da2, self.da1, self.grid,
                                               self.h0,  self.h1c, self.h1h, self.h2c, self.h2h,
                                               self.h11, self.h12, self.h21, self.h22)
        
        self.vlasov_solver.set_moments(self.nc, self.uc, self.ec, self.ac,
                                       self.nh, self.uh, self.eh, self.ah)
        
        
        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.k.getSizes(), self.b.getSizes()], 
                                            context=self.vlasov_solver,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
        
        
        # create nonlinear predictor solver
        self.snes = PETSc.SNES().create()
        self.snes.setType('ksponly')
        self.snes.setFunction(self.vlasov_solver.function_snes_mult, self.b)
        self.snes.setJacobian(self.updateVlasovJacobian, self.Jmf)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('gmres')
        self.snes.getKSP().getPC().setType('none')
        
        
        # create Poisson matrix and object
        self.poisson_matrix = self.dax.createMat()
        self.poisson_matrix.setUp()
        self.poisson_matrix.setNullSpace(self.p_nullspace)
        
        self.poisson_solver = PETScPoissonSolver(self.dax, self.grid.nx, self.grid.hx, self.charge)
        self.poisson_solver.formMat(self.poisson_matrix)
        
        self.poisson_mf = PETSc.Mat().createPython([self.pc_int.getSizes(), self.pb.getSizes()], 
                                                   context=self.poisson_solver,
                                                   comm=PETSc.COMM_WORLD)
        self.poisson_mf.setUp()
        
        
        # create linear Poisson solver
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setTolerances(rtol=1E-13)
        self.poisson_ksp.setOperators(self.poisson_mf, self.poisson_matrix)
        self.poisson_ksp.setType('cg')
#         self.poisson_ksp.setType('bcgs')
#         self.poisson_ksp.setType('ibcgs')
        self.poisson_ksp.getPC().setType('hypre')
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Run script initialisation done.")
            print("")
    
    
    def __enter__(self):
        return self
    
    
    def __exit__(self,ext_type,exc_value,traceback):
        self.poisson_ksp.destroy()
        self.snes.destroy()
        
        self.poisson_mf.destroy()
        self.Jmf.destroy()
        
        del self.poisson_solver
        del self.vlasov_solver
        
        super().destroy()
    
    
    def updateVlasovJacobian(self, snes, X, J, P):
        if J != P:
            self.vlasov_solver.formJacobian(P)
    
    
    def run(self):
        for itime in range(1, self.grid.nt+1):
            current_time = self.grid.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # update history
            self.make_history()
            
            # compute initial guess
            self.initial_guess4()
            
            # calculate external field and copy to solver
            self.calculate_external4(itime)
            
            # update current solution in solver
            self.vlasov_solver.update_previous4()
            
            # nonlinear solve
            i = 0
            pred_norm = self.calculate_residual4()
            while True:
                i+=1
                
                self.k.copy(self.kh)
                
                self.snes.solve(None, self.k)
                
                self.calculate_moments4(output=False)
                self.vlasov_solver.update_previous4()
                
                prev_norm = pred_norm
                pred_norm = self.calculate_residual4()
                phisum1 = self.p1_int.sum()
                phisum2 = self.p2_int.sum()

                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlinear Solver: %5i GMRES  iterations, residual = %24.16E" % (self.snes.getLinearSolveIterations(),  pred_norm) )
                    print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.p1_niter, phisum1))
                    print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.p2_niter, phisum2))
                
                if (pred_norm > prev_norm and i > 1) or pred_norm < self.cfg['solver']['petsc_snes_atol'] or i >= self.cfg['solver']['petsc_snes_max_iter']:
                    if pred_norm > prev_norm:
                        self.kh.copy(self.k)
                        self.calculate_moments4(output=False)
                    
                    break
            
            # compute final distribution function, potential and moments
#             self.fh.copy(self.f)
            self.fc.axpy(self.b1 * self.grid.ht, self.k1)
            self.fc.axpy(self.b2 * self.grid.ht, self.k2)
            
            self.calculate_moments(output=False)
            phisum = self.pc_int.sum()
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Poisson   Solver: %5i CG     iterations, sum(phi) = %24.16E" % (self.poisson_ksp.getIterationNumber(), phisum)    )
            
            # save to hdf5
            self.save_to_hdf5(itime)
        
        # flush all data
        self.hdf5_viewer.flush()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('-c', metavar='<cfg_file>', type=str,
                        help='Configuration File')
    parser.add_argument('-i', metavar='<runid>', type=str, default="",
                        help='Run ID')
    
    args = parser.parse_args()
    
    with petscVP1Dmatrixfree(args.c, args.i) as petscvp:
        petscvp.run()
    
