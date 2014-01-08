'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse, time

from petsc4py import PETSc


from run_nonlinear_matrixfree_split import petscVP1Dmatrixfree


class petscVP1Ddamped(petscVP1Dmatrixfree):
    '''
    PETSc/Python Vlasov Poisson GMRES Solver in 1D.
    '''

    def calculate_external(self, t):
        (xs, xe), = self.da1.getRanges()
        
        if self.external != None:
            p_ext_arr = self.dax.getVecArray(self.p_ext)
            p_arr     = self.dax.getVecArray(self.p)
            
            for i in range(xs, xe):
                p_ext_arr[i] = (self.external(self.xGrid[i], t) - 1.) * p_arr[i] 
            
            # remove average
            phisum = self.p_ext.sum()
            phiave = phisum / self.nx
            self.p_ext.shift(-phiave)
    
        self.copy_pext_to_h()



    def run(self):

        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # compute initial guess
            self.initial_guess()
            
            # calculate external field and copy to solver
            self.calculate_external(current_time)
            self.vlasov_solver.update_previous(self.f, self.p, self.p_ext, self.n, self.u, self.e)
            
            # nonlinear solve
            i = 0
            pred_norm = self.calculate_residual()
            while True:
                i+=1
                
                if i == 1:
                    self.snes.getKSP().setTolerances(rtol=1E-5)
                if i == 2:
                    self.snes.getKSP().setTolerances(rtol=1E-4)
                if i == 3:
                    self.snes.getKSP().setTolerances(rtol=1E-3)
                if i == 4:
                    self.snes.getKSP().setTolerances(rtol=1E-3)
                
                
                self.f.copy(self.fh)
                
                self.calculate_external(current_time)
                
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
    
    petscvp = petscVP1Ddamped(args.c, args.i)
    petscvp.run()
    
