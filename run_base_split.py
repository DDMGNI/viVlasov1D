'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys
import petsc4py
petsc4py.init(sys.argv)

# import numpy as np

from petsc4py import PETSc

# from vlasov.core.config     import Config
# from vlasov.toolbox.maxwell import maxwellian

from run_base import petscVP1Dbase


class petscVP1Dbasesplit(petscVP1Dbase):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile, runid=None, cfg=None):
        '''
        Constructor
        '''
        
        super().__init__(cfgfile, runid, cfg)
        
        
        # create matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.fc.getSizes(), self.fb.getSizes()], 
                                            comm=PETSc.COMM_WORLD)
        

