'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse, time

from petsc4py import PETSc

from run_base_split_db import petscVP1DbasesplitDB


class petscVP1Drunscript(petscVP1DbasesplitDB):
    '''
    PETSc/Python Vlasov Poisson GMRES Solver in 1D.
    '''


    def __init__(self, cfgfile, runid=None, cfg=None):
        super().__init__(cfgfile, runid, cfg)
        
        # create solver objects
        self.vlasov_solver = self.vlasov_object.PETScVlasovSolver(
                                               self.da2, self.da1, self.grid,
                                               self.h0,  self.h1c, self.h1h, self.h2c, self.h2h,
                                               charge=self.charge, coll_freq=self.coll_freq)
        
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
#         self.poisson_ksp.getPC().setType('hypre')
        self.poisson_ksp.getPC().setType('lu')
        self.poisson_ksp.getPC().setFactorSolverPackage('superlu_dist')
        
        
        # initialise data structure
        hc = self.da1.createGlobalVec()
        hc.set(0.)
        self.h0.copy(hc)
        self.h1c.copy(hc)
        self.h2c.copy(hc)
        
        k_arr = self.da2.getGlobalArray(self.k)
        f_arr = self.da1.getGlobalArray(self.fc)
        g_arr = self.da1.getGlobalArray(self.gc)
        h_arr = self.da1.getGlobalArray(hc)
        
        self.arakawa.arakawa_J1_timestep(f_arr, g_arr, h_arr)
                
        k_arr[:, :, 0] = f_arr[:, :]
        k_arr[:, :, 1] = g_arr[:, :]
        
        hc.destroy()
        
        
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
            self.make_history_db()
            
            # compute initial guess
#             self.initial_guess_db()
            
            # calculate external field and copy to solver
            self.calculate_external(itime)
            
            # update current solution in solver
            self.vlasov_solver.update_previous_db(self.fc, self.gc)
            
            # nonlinear solve
            i = 0
            pred_norm = self.calculate_residual_db()
            while True:
                i+=1
                
                self.k.copy(self.kh)
                
                self.snes.solve(None, self.k)
                
                self.calculate_moments_db(output=False)
                self.vlasov_solver.update_previous_db(self.fc, self.gc)
                
                prev_norm = pred_norm
                pred_norm = self.calculate_residual_db()
                phisum = self.pc_int.sum()

                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlinear Solver: %5i GMRES  iterations, residual = %24.16E" % (self.snes.getLinearSolveIterations(),  pred_norm) )
                    print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.poisson_ksp.getIterationNumber(), phisum))
                
                if (pred_norm > prev_norm and i > 1) or pred_norm < self.cfg['solver']['petsc_snes_atol'] or i >= self.cfg['solver']['petsc_snes_max_iter']:
                    if pred_norm > prev_norm:
                        self.kh.copy(self.k)
                        self.calculate_moments_db(output=False)
                    
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
    
