'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse, time

from petsc4py import PETSc
from run_base_split import petscVP1Dbasesplit

# from vlasov.solvers.vlasov.PETScNLVlasovArakawaJ1       import PETScVlasovSolver
# from vlasov.solvers.vlasov.PETScNLVlasovArakawaJ2       import PETScVlasovSolver
# from vlasov.solvers.vlasov.PETScNLVlasovArakawaJ4       import PETScVlasovSolver
# from vlasov.solvers.vlasov.PETScNLVlasovSimpson         import PETScVlasovSolver

# from vlasov.solvers.vlasov.PETScNLVlasovTriangle1       import PETScVlasovSolver
from vlasov.solvers.vlasov.PETScNLVlasovTriangle2       import PETScVlasovSolver


# solver_package = 'superlu_dist'
solver_package = 'mumps'
# solver_package = 'pastix'


class petscVP1Dlu(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson LU Solver in 1D.
    '''

    def updateVlasovJacobian(self, snes, X, J, P):
#         self.vlasov_solver.update_delta(X)
        self.vlasov_solver.formJacobian(J)
        
        if J != P:
            self.vlasov_solver.formJacobian(P)
        
    
    def __init__(self, cfgfile, runid):
        super().__init__(cfgfile, runid)

#         OptDB = PETSc.Options()
        
#         OptDB.setValue('ksp_monitor',  '')
#         OptDB.setValue('snes_monitor', '')
        
#        OptDB.setValue('log_info',    '')
#        OptDB.setValue('log_summary', '')
        

        # initialise Jacobian
        self.J = self.da1.createMat()
        self.J.setOption(self.J.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.J.setUp()
        

        # create solver objects
        self.vlasov_solver = PETScVlasovSolver(self.da1, self.dax,
                                               self.h0, self.vGrid,
                                               self.nx, self.nv, self.ht, self.hx, self.hv,
                                               self.charge, coll_freq=self.coll_freq)
        
        
        # update solution history
        self.vlasov_solver.update_history(self.f, self.p, self.p_ext, self.n, self.u, self.e)


        # create nonlinear solver
        self.snes = PETSc.SNES().create()
        self.snes.setFunction(self.vlasov_solver.function_snes_mult, self.fb)
        self.snes.setJacobian(self.updateVlasovJacobian, self.J)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('preonly')
        self.snes.getKSP().getPC().setType('lu')
        self.snes.getKSP().getPC().setFactorSolverPackage(solver_package)
        
        
        del self.poisson_ksp
        
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_A)
        self.poisson_ksp.setType('preonly')
        self.poisson_ksp.getPC().setType('lu')
        self.poisson_ksp.getPC().setFactorSolverPackage(solver_package)
        
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # calculate external field and copy to matrix, jacobian and function
            self.calculate_external(current_time)
            self.vlasov_solver.update_previous(self.f, self.p, self.p_ext, self.n, self.u, self.e)
            
            # compute initial guess
            self.initial_guess()
            
            
            i = 0
            pred_norm = self.calculate_residual()
            while True:
                i+=1
                
                self.f.copy(self.fh)
                
                self.vlasov_solver.update_previous(self.f, self.p, self.p_ext, self.n, self.u, self.e)
                self.snes.solve(None, self.f)
                
                self.calculate_moments(output=False)
                self.vlasov_solver.update_previous(self.f, self.p, self.p_ext, self.n, self.u, self.e)
                
                prev_norm = pred_norm
                pred_norm = self.calculate_residual()
                phisum = self.p.sum()
                
                
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("  Nonlinear Solver:                          residual = %24.16E" % (pred_norm) )
                    print("                                             sum(phi) = %24.16E" % (phisum))
                
                if pred_norm > prev_norm or pred_norm < self.cfg['solver']['petsc_snes_atol'] or i >= self.cfg['solver']['petsc_snes_max_iter']:
                    if pred_norm > prev_norm:
                        self.fh.copy(self.f)
                        self.calculate_moments(output=False)
                    
                        if PETSc.COMM_WORLD.getRank() == 0:
                            print()
                            print("Solver not converging...   %i" % (self.snes.getConvergedReason()))
                            print()
                    
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
    
    petscvp = petscVP1Dlu(args.c, args.i)
    petscvp.run()
    
