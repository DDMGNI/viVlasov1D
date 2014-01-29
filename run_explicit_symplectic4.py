'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse
import sys, time
import numpy as np

from petsc4py import PETSc

from run_base_split import petscVP1Dbasesplit

from vlasov.explicit.PETScArakawaSymplectic import PETScArakawaSymplectic


class petscVP1Drunscript(petscVP1Dbasesplit):
    '''
    PETSc/Python Vlasov Poisson LU Solver in 1D.
    '''

    def __init__(self, cfgfile, runid):
        super().__init__(cfgfile, runid)
        
        self.arakawa_symplectic = PETScArakawaSymplectic(self.da1, self.dax,
                                                         self.h0, self.vGrid,
                                                         self.nx, self.nv, self.ht, self.hx, self.hv)
        
        
    def run(self):
        
        fac2 = 2.**(1./3.)
         
        c1 = 0.5 / ( 2. - fac2 )
        c2 = c1  * ( 1. - fac2 )
        c3 = c2
        c4 = c1
         
        d1 = 1. / ( 2. - fac2 )
        d2 = - d1 * fac2
        d3 = d1
         
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("it = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                self.time.setValue(0, current_time)
            
            # solve
            self.arakawa_symplectic.kinetic(self.f, c1)
            self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.f, self.h1, d1)
             
            self.arakawa_symplectic.kinetic(self.f, c2)
            self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.f, self.h1, d2)
             
            self.arakawa_symplectic.kinetic(self.f, c3)
            self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.f, self.h1, d3)
             
            self.arakawa_symplectic.kinetic(self.f, c4)
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
    
