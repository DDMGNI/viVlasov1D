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
from vlasov.plot import PlotEnergy


class replay(object):
    '''
    
    '''


    def __init__(self, hdf5_file, first=-1, last=-1, vMax=0., linear=False):
        '''
        Constructor
        '''
        
        self.first = first
        self.last  = last
        
        self.hdf5 = h5py.File(hdf5_file, 'r')
        
        # read config file from HDF5 and create config object
#        cfg_str = self.hdf5['runcfg'][:][0]
#        
#        cfg_io = StringIO.StringIO(cfg_str.strip())
#        cfg    = core.Config(cfg_io)
#        cfg_io.close()
        
        self.grid         = Grid                (hdf5_in=self.hdf5, replay=True)
        self.potential    = Potential           (self.grid, hdf5_in=self.hdf5, replay=True,
                                                 poisson_const=-1.)
        self.hamiltonian  = Hamiltonian         (self.grid, hdf5_in=self.hdf5, replay=True, linear=linear)
        self.distribution = DistributionFunction(self.grid, hdf5_in=self.hdf5, replay=True)
        
        self.potential.read_from_hdf5(0)
        self.distribution.read_from_hdf5(0)
        self.hamiltonian.read_from_hdf5(0)
        
        self.plot = PlotEnergy(self.grid, self.distribution, self.hamiltonian, self.potential, first, last, vMax)
        
    
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
                        help='Run HDF5 File')
    parser.add_argument('-pi', metavar='i', type=int, default=1,
                        help='plot pi\'th frame')    
    parser.add_argument('-fi', metavar='i', type=int, default=-1,
                        help='first time index')
    parser.add_argument('-li', metavar='i', type=int, default=-1,
                        help='last time index')
    parser.add_argument('-v', metavar='f', type=float, default=0.0,
                        help='limit velocity domain to +/-v')
    parser.add_argument('-linear', metavar='b', type=bool, default=False,
                        help='use linear diagnostics')
    
    args = parser.parse_args()
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = replay(args.hdf5_file, args.fi, args.li, args.v, args.linear)
    pyvp.run(args.pi)
    
    print
    print("Replay finished.")
    print
    
