'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

#import StringIO
import argparse
import h5py

from vlasov.core import DistributionFunction, Grid, Hamiltonian, Potential
from vlasov.plot import PlotEnergy


class replay(object):
    '''
    
    '''


    def __init__(self, hdf5_file, nPlot=-1):
        '''
        Constructor
        '''
        
        self.nPlot  = nPlot
        
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
        self.hamiltonian  = Hamiltonian         (self.grid, hdf5_in=self.hdf5, replay=True)
        self.distribution = DistributionFunction(self.grid, hdf5_in=self.hdf5, replay=True)
        
        self.potential.read_from_hdf5(0)
        self.distribution.read_from_hdf5(0)
        self.hamiltonian.read_from_hdf5(0)
        
        self.plot = PlotEnergy(self.grid, self.distribution, self.hamiltonian, self.potential, nPlot)
        
    
    def __del__(self):
        if self.hdf5 != None:
            self.hdf5.close()
        
    
    def run(self, ptime):
        if ptime != 0:
            for itime in range(1, self.grid.nt+1):
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
    parser.add_argument('-li', metavar='i', type=int, default=-1,
                        help='last time index')
    
    args = parser.parse_args()
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = replay(args.hdf5_file, args.li)
    pyvp.run(args.pi)
    
    print
    print("Replay finished.")
    print
    