'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import matplotlib
#matplotlib.use('Cairo')
matplotlib.use('AGG')
#matplotlib.use('PDF')

#import StringIO
import argparse
import h5py

from vlasov.core import DistributionFunction, Grid, Hamiltonian, Potential
from vlasov.plot import PlotTimetraces


class timetraces(object):
    '''
    Plots timetraces of the errors of the total particle number, momentum, energy, entropy,
    the L1, L2, L3, L4, L6, L8 norms as well as the distribution function at the final timestep.
    
    The script is invoked with
    
        ``python diag_timetraces.py -f <findex> -l <lindex> -v <vmax> -i <hdf5_file>``
    
    -f <findex>     is the index of the first timestep to plot (*default*: 0)
    -l <lindex>     is the index of the last timestep to plot (*default*: nt)
    -v <vmax>       limits the v range to plot in the contour plot of the distribution function
    -i <hdf5_file>  specifies the data file to read
    
    '''


    def __init__(self, hdf5_file, first=-1, last=-1, vMax=0.):
        '''
        Constructor
        '''
        
        self.first = first
        self.last  = last
        
        self.hdf5 = h5py.File(hdf5_file, 'r')
        self.grid = Grid().load_from_hdf5(self.hdf5)
        
        self.potential    = Potential           (self.grid, self.hdf5)
        self.hamiltonian  = Hamiltonian         (self.grid, hdf5=self.hdf5)
        self.distribution = DistributionFunction(self.grid, hdf5=self.hdf5)
        
        self.potential.read_from_hdf5(0)
        self.distribution.read_from_hdf5(0)
        self.hamiltonian.read_from_hdf5(0)
        
        self.plot = PlotTimetraces(self.grid, self.distribution, self.hamiltonian, self.potential, first, last, vMax)
        
    
    def __del__(self):
        if self.hdf5 != None:
            self.hdf5.close()
        
    
    def run(self, ptime):
        if ptime != 0:
            for itime in range(1, self.last+1):
                print("it = %5i" % (itime))
                
                self.potential.read_from_hdf5(itime)
                self.distribution.read_from_hdf5(itime)
                self.hamiltonian.read_from_hdf5(itime)
                
                self.plot.add_timepoint()
                
                if itime == ptime:
                    self.plot.update()
                    break
        
        self.plot.save_plots()
        self.plot.update(final=True)
        
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Vlasov-Poisson Solver in 1D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='HDF5 data file')
    parser.add_argument('-f', metavar='<findex>', type=int, default=-1,
                        help='first time index')
    parser.add_argument('-l', metavar='<lindex>', type=int, default=-1,
                        help='last time index')
    parser.add_argument('-v', metavar='<vmax>', type=float, default=0.0,
                        help='limit velocity domain to +/-vmax')
    
    args = parser.parse_args()
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = timetraces(args.hdf5_file, args.f, args.l, args.v)
    pyvp.run(args.l)
    
    print
    print("Replay finished.")
    print
    
