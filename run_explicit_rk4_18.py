'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse
import sys, time
import numpy as np

from petsc4py import PETSc

from run_base_split import petscVP1Dbasesplit

from vlasov.explicit.PETScArakawaRungeKutta import PETScArakawaRungeKutta


class petscVP1Drunscript(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson LU Solver in 1D.
    '''

    def __init__(self, cfgfile, runid):
        super().__init__(cfgfile, runid)
        
        self.arakawa_rk = PETScArakawaRungeKutta(self.da1, self.dax,
                                                 self.h0, self.vGrid,
                                                 self.nx, self.nv, self.ht, self.hx, self.hv)
        
        
    def run(self):
        
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("it = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                self.time.setValue(0, current_time)
            
            # solve
            self.arakawa_rk.rk18_J4(self.f, self.h1)
            self.calculate_moments(output=False)
            
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
    
