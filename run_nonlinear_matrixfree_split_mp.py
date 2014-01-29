'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse, time
# import pstats, cProfile

from petsc4py import PETSc

from run_base_split import petscVP1Dbasesplit


class petscVP1Drunscript(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson GMRES Solver in 1D.
    '''

    def __init__(self, cfgfile="", runid="", cfg=None):
        super().__init__(cfgfile, runid, cfg)
        
#         OptDB = PETSc.Options()
        
#         OptDB.setValue('snes_ls', 'basic')

#         OptDB.setValue('ksp_monitor',  '')
#         OptDB.setValue('snes_monitor', '')
        
#         OptDB.setValue('log_info',    '')
#         OptDB.setValue('log_summary', '')
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Creating solver objects.")
    
        # create solver objects
        self.vlasov_solver = self.vlasov_object.PETScVlasovSolver(
                                               self.da1, self.grid,
                                               self.h0, self.h1c, self.h1h, self.h2c, self.h2h,
                                               self.charge, coll_freq=self.coll_freq)
        
        self.vlasov_solver.set_moments(self.nc, self.uc, self.ec, self.ac,
                                       self.nh, self.uh, self.eh, self.ah)
        
        
        # initialise matrixfree Jacobian
        self.Jmf.setPythonContext(self.vlasov_solver)
        self.Jmf.setUp()
        
        
        # create nonlinear predictor solver
        self.snes = PETSc.SNES().create()
        self.snes.setType('ksponly')
        self.snes.setFunction(self.vlasov_solver.function_snes_mult, self.fb)
        self.snes.setJacobian(self.updateVlasovJacobian, self.Jmf)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('gmres')
        self.snes.getKSP().getPC().setType('none')
        
        
        # create Poisson matrix and object
        self.poisson_matrix = self.dax.createMat()
        self.poisson_matrix.setUp()
        self.poisson_matrix.setNullSpace(self.p_nullspace)
        
        self.poisson_solver = self.poisson_object.PETScPoissonSolver(self.dax, self.grid.nx, self.grid.hx, self.charge)
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
            
            # calculate external field and copy to solver
            self.calculate_external(current_time)
            
            # compute initial guess
            self.initial_guess()
            
            # update current solution in solver
            self.vlasov_solver.update_previous(self.fc)
            
            # nonlinear solve
            i = 0
            pred_norm = self.calculate_residual()
            while True:
                i+=1
                
                self.fc.copy(self.fl)
                
                self.snes.solve(None, self.fc)
                
                self.calculate_moments(output=False)
                self.vlasov_solver.update_previous(self.fc)
                
                prev_norm = pred_norm
                pred_norm = self.calculate_residual()
                phisum = self.pc_int.sum()

                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlinear Solver: %5i GMRES  iterations, residual = %24.16E" % (self.snes.getLinearSolveIterations(), pred_norm) )
                    print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.poisson_ksp.getIterationNumber(), phisum))
                
                if (pred_norm > prev_norm and i > 1) or pred_norm < self.cfg['solver']['petsc_snes_atol'] or i >= self.cfg['solver']['petsc_snes_max_iter']:
                    if pred_norm > prev_norm:
                        self.fl.copy(self.fc)
                        self.calculate_moments(output=False)
                    
                    break
            
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
    
    with petscVP1Drunscript(args.c, args.i) as petscvp:
        petscvp.run()
    
#     cProfile.runctx("petscvp.run()", globals(), locals(), "Profile.prof")
#       
#     s = pstats.Stats("Profile.prof")
#     s.strip_dirs().sort_stats("time").print_stats()
    
