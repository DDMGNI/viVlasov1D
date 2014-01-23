'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse, time
import pstats, cProfile

from petsc4py import PETSc

# from vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4TensorPETSc import PETScVlasovSolver
from vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4TensorSciPy import PETScVlasovSolver

from vlasov.solvers.poisson.PETScPoissonSolver4  import PETScPoissonSolver

from run_base_split import petscVP1Dbasesplit


class petscVP1Dmatrixfree(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson GMRES Solver in 1D.
    '''


    def __init__(self, cfgfile, runid):
        super().__init__(cfgfile, runid)
        
        OptDB = PETSc.Options()
        
        
        OptDB.setValue('snes_type', 'newtonls')
        OptDB.setValue('snes_linesearch_type', 'basic')
        OptDB.setValue('snes_linesearch_monitor', '')
        
#         OptDB.setValue('ksp_monitor',  '')
        OptDB.setValue('snes_monitor', '')
        
#         OptDB.setValue('log_info',    '')
#         OptDB.setValue('log_summary', '')
        
        
        # create solver objects
        self.vlasov_solver = PETScVlasovSolver(self.da1, self.grid,
                                               self.h0, self.h1c, self.h1h, self.h2c, self.h2h,
                                               self.charge, coll_freq=self.coll_freq)
        
        self.vlasov_solver.set_moments(self.nc, self.uc, self.ec, self.ac,
                                       self.nh, self.uh, self.eh, self.ah)
        
        
        # initialise matrixfree Jacobian
        self.Jmf.setPythonContext(self.vlasov_solver)
        self.Jmf.setUp()
        
        
        # create nonlinear predictor solver
        self.snes = PETSc.SNES().create()
        self.snes.setFunction(self.vlasov_solver.function_snes_mult, self.fb)
        self.snes.setJacobian(self.updateVlasovJacobian, self.Jmf)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('gmres')
        self.snes.getKSP().getPC().setType('none')
        
        
        
        del self.poisson_ksp
        del self.poisson_solver
            
        self.poisson_solver = PETScPoissonSolver(self.dax, self.grid.nx, self.grid.hx, self.charge)
        self.poisson_solver.formMat(self.poisson_matrix)
        
        self.poisson_mf = PETSc.Mat().createPython([self.pc_int.getSizes(), self.pb.getSizes()], 
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
        self.calculate_moments(output=False, f=X)
        self.vlasov_solver.update_previous(X)
        
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
#             self.vlasov_corrector.update_history(self.fc)
            
            # calculate external field and copy to solver
            self.calculate_external(current_time)
            
            # compute initial guess
            self.initial_guess()
            
            # nonlinear solve
            self.snes.solve(None, self.fc)
            
            # some output
            prev_norm = self.calculate_residual()
            phisum = self.pc_int.sum()
 
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Nonlinear Solver: %5i Newton iterations, residual = %24.16E" % (self.snes.getIterationNumber(),        prev_norm) )
                print("                    %5i GMRES  iterations                    " % (self.snes.getLinearSolveIterations()            ) )
                print("                    %5i CG     iterations, sum(phi) = %24.16E" % (self.poisson_ksp.getIterationNumber(), phisum   ) )
                
            
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
#     petscvp.run()

    cProfile.runctx("petscvp.run()", globals(), locals(), "Profile_Tensor.prof")
      
    s = pstats.Stats("Profile_Tensor.prof")
    s.strip_dirs().sort_stats("time").print_stats()


    
