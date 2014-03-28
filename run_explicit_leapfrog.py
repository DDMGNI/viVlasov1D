'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse
import sys, time
import numpy as np

from petsc4py import PETSc

from run_base_split import petscVP1Dbasesplit

from vlasov.solvers.explicit.PETScArakawaLeapfrog   import PETScArakawaLeapfrog
from vlasov.solvers.explicit.PETScArakawaRungeKutta import PETScArakawaRungeKutta


class petscVP1Drunscript(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson LU Solver in 1D.
    '''

    def __init__(self, cfgfile, runid):
        super().__init__(cfgfile, runid)
        
        self.arakawa_lf = PETScArakawaLeapfrog(self.da1, self.dax,
                                               self.h0, self.vGrid,
                                               self.nx, self.nv, self.ht, self.hx, self.hv)
        
        self.arakawa_rk = PETScArakawaRungeKutta(self.da1, self.dax,
                                                 self.h0, self.vGrid,
                                                 self.nx, self.nv, -self.ht, self.hx, self.hv)
        
        
        self.f.copy(self.fh)
        
        self.arakawa_rk.rk18_J4(self.f, self.h1)
        self.calculate_moments(output=False)
        
#         self.calculate_external(-self.ht)
#         self.arakawa_lf.update_external(self.p_ext)
        self.arakawa_lf.update_history(self.f, self.h1)
        
        
        self.fh.copy(self.f)
        self.calculate_moments(output=False)
        
#         self.calculate_external(0.)
#         self.arakawa_lf.update_external(self.p_ext)
        self.arakawa_lf.update_history(self.f, self.h1)
        
    
    def run(self):
        
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("it = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
#                 print
                self.time.setValue(0, current_time)
            
            # calculate external field and copy to solver
#             self.calculate_external(current_time)
#             self.arakawa_lf.update_external(self.p_ext)
            
            # solve
            self.arakawa_lf.leapfrog2(self.f)
#             self.arakawa_lf.leapfrog4(self.f)
            self.calculate_moments(output=False)
            
            # update history
            self.arakawa_lf.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('-c', metavar='<cfg_file>', type=str,
                        help='Configuration File')
    parser.add_argument('-i', metavar='<runid>', type=str, default="",
                        help='Run ID')
    
    args = parser.parse_args()
    
    petscvp = petscVP1Drunscript(args.c, args.i)
    petscvp.run()
    
