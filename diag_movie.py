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
from vlasov.plot import PlotMovie


class movie(object):
    '''
    Creates a movie from HDF5 showing the distribution function
    and timetraces of the errors in the particle number, energy,
    entropy and L2 norm.
    '''


    def __init__(self, hdf5_file, ntMax=0, nTime=0, iStart=0, nPlot=1, vMin=None, vMax=None, cMax=False, cFac=1.0):
        '''
        Constructor
        '''
        
        self.iStart = iStart
        self.nPlot  = nPlot
        
        self.hdf5 = h5py.File(hdf5_file, 'r')
        self.grid = Grid().load_from_hdf5(self.hdf5)
        
        self.potential    = Potential           (self.grid, self.hdf5)
        self.hamiltonian  = Hamiltonian         (self.grid, hdf5=self.hdf5)
        self.distribution = DistributionFunction(self.grid, hdf5=self.hdf5)
        
        self.potential.read_from_hdf5(iStart)
        self.distribution.read_from_hdf5(iStart)
        self.hamiltonian.read_from_hdf5(iStart)

        
        if ntMax > 0 and ntMax < self.grid.nt:
            self.nt = ntMax
        else:
            self.nt = self.grid.nt
        
        self.plot = PlotMovie(self.grid, self.distribution, self.hamiltonian, self.potential,
                              nTime=nTime, nPlot=nPlot, ntMax=self.nt,
                              vMin=vMin, vMax=vMax, cMax=cMax, cFac=cFac, write=True)
        
    
    def __del__(self):
        if self.hdf5 != None:
            self.hdf5.close()
        
    
    def update(self, itime, final=False):
        self.potential.read_from_hdf5(itime)
        self.distribution.read_from_hdf5(itime)
        self.hamiltonian.read_from_hdf5(itime)
        
        if itime > 0:
            self.plot.add_timepoint()
        
        return self.plot.update(final=final)
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            print(itime)
            self.update(itime, final=(itime == self.nt))
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vlasov-Poisson Solver in 1D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='Run HDF5 File')
    parser.add_argument('-nt', metavar='i', type=int, default=0,
                        help='plot i points in time traces')
    parser.add_argument('-np', metavar='i', type=int, default=1,
                        help='plot every i\'th frame')
    parser.add_argument('-ntmax', metavar='i', type=int, default=0,
                        help='limit to i points in time')
    parser.add_argument('-v', metavar='f', type=float, default=None,
                        help='limit velocity domain to +/-v')
    parser.add_argument('-vmin', metavar='f', type=float, default=None,
                        help='set lower limit of velocity domain to vmin')
    parser.add_argument('-vmax', metavar='f', type=float, default=None,
                        help='set upper limit of velocity domain to vmax')
    parser.add_argument('-cmax', metavar='b', type=bool, default=False,
                        help='use max values of simulation in contour plots')
    parser.add_argument('-cfac', metavar='f', type=float, default=1.0,
                        help='multiply max value of initial data in contour plots')
    parser.add_argument('-fps', metavar='i', type=int, default=1,
                        help='frames per second')    
    
    args = parser.parse_args()
    
    vMin = None
    vMax = None
    
    if args.v is not None:
        vMin = -args.v
        vMax = +args.v
    else:
        if args.vmin is not None:
            vMin = args.vmin 
        if args.vmax is not None:
            vMax = args.vmax 
     
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = movie(args.hdf5_file, ntMax=args.ntmax, nTime=args.nt, nPlot=args.np,
                 vMin=vMin, vMax=vMax, cMax=args.cmax, cFac=args.cfac)
    
    pyvp.run()
    
    print
    print("Replay finished.")
    print
    
