'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import matplotlib
#matplotlib.use('Cairo')
matplotlib.use('AGG')
#matplotlib.use('PDF')

import argparse
import h5py

from vlasov.core import DistributionFunction, Grid, Hamiltonian, Potential
from vlasov.plot import PlotBGK


class movie(object):
    '''
    Creates a movie from HDF5 showing the distribution function
    and timetraces of the errors in the particle number, energy,
    entropy and L2 norm.
    '''


    def __init__(self, hdf5_file, iPlot=1):
        '''
        Constructor
        '''
        
        self.hdf5 = h5py.File(hdf5_file, 'r')
        self.grid = Grid().load_from_hdf5(self.hdf5)
        
        if iPlot > 0 and iPlot < self.grid.nt:
            self.iPlot = iPlot
        else:
            self.iPlot = self.grid.nt
        
        self.potential    = Potential           (self.grid, self.hdf5)
        self.hamiltonian  = Hamiltonian         (self.grid, hdf5=self.hdf5)
        self.distribution = DistributionFunction(self.grid, hdf5=self.hdf5)
        
        self.potential.read_from_hdf5(iPlot)
        self.distribution.read_from_hdf5(iPlot)
        self.hamiltonian.read_from_hdf5(iPlot)
        
        self.plot = PlotBGK(self.grid, self.distribution, self.hamiltonian, self.potential,
                              iTime=iPlot, write=True)
        
    
    def __del__(self):
        if self.hdf5 != None:
            self.hdf5.close()
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vlasov-Poisson Solver in 1D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='Run HDF5 File')
    parser.add_argument('-iplot', metavar='i', type=int, default=0,
                        help='plot i-th timestep')

    
    args = parser.parse_args()
    
    print()
    print("Replay run with " + args.hdf5_file)
    print()
    
    pyvp = movie(args.hdf5_file, iPlot=args.iplot)
    
