'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

#import StringIO
import argparse
import os, sys
import numpy as np
import h5py


#sys.path.append(os.getcwd())


from vlasov.core import DistributionFunction, Grid, Hamiltonian, Potential
from vlasov.plot import PlotSpecies


class replay(object):
    '''
    Interactive replay plotting the distribution function, the Hamiltonian,
    density, potential and timetraces of kinetic and potential energy 
    as well as the errors in the particle number, energy, entropy and L2 norm.
    
    '''


    def __init__(self, hdf5_file, nPlot=1, iStart=0):
        '''
        Constructor
        '''
        
        self.iStart = iStart
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
        self.hamiltonian  = Hamiltonian         (self.grid, hdf5=self.hdf5)
        self.distribution = DistributionFunction(self.grid, hdf5_in=self.hdf5, replay=True)
        
        self.potential.read_from_hdf5(iStart)
        self.distribution.read_from_hdf5(iStart)
        self.hamiltonian.read_from_hdf5(iStart)
        
        self.plot = PlotSpecies(self.grid, self.distribution, self.hamiltonian, self.potential,
                                self.grid.nt, iStart, nPlot)
        
        self.plot.save_plots()
        
    
    def __del__(self):
        if self.hdf5 != None:
            self.hdf5.close()
        
    
    def init(self, iStart=0):
        self.update(iStart)
    
    
    def update(self, itime, final=False):
        self.potential.read_from_hdf5(itime)
        self.distribution.read_from_hdf5(itime)
        self.hamiltonian.read_from_hdf5(itime)
        
        print("it = %5i" % (itime))
        
        
        if itime > 0:
#            Ekin  = self.hamiltonian.Ekin
#            Ekin0 = self.hamiltonian.Ekin0
#            Epot  = self.hamiltonian.Epot  - self.potential.E
#            Epot0 = self.hamiltonian.Epot0 - self.potential.E0
#            Epot_f  = self.hamiltonian.Epot  / 2.
#            Epot0_f = self.hamiltonian.Epot0 / 2.
#            Epot_p  = self.potential.E
#            Epot0_p = self.potential.E0
#            
#            dEkin = Ekin - Ekin0
#            dEpot = Epot - Epot0
#            dEpot_f = Epot_f - Epot0_f
#            dEpot_p = Epot_p - Epot0_p
#            
#            E    = Ekin  + Epot
#            E0   = Ekin0 + Epot0
#            E_f  = Ekin  + Epot_f
#            E0_f = Ekin0 + Epot0_f
#            E_p  = Ekin  + Epot_p
#            E0_p = Ekin0 + Epot0_p
#            
#            E_err   = (E   - E0  ) / E0
#            E_err_f = (E_f - E0_f) / E0_f
#            E_err_p = (E_p - E0_p) / E0_p
#            
#            print
#            print("      dEpot/dEkin = %24.16E,  dEpot(f)/dEkin = %24.16E,  dEpot(phi)/dEkin = %24.16E" % (dEpot/dEkin, dEpot_f/dEkin, dEpot_p/dEkin))
#            print("      E_pot       = %24.16E,  E_pot(f)       = %24.16E,  E_pot(phi)       = %24.16E" % (Epot, Epot_f, Epot_p))
#            print("      E_err       = %24.16E,  E_err(f)       = %24.16E,  E_err(phi)       = %24.16E" % (E_err, E_err_f, E_err_p))
            
            
            self.plot.add_timepoint()
        
        return self.plot.update(final=final)
    
    
    def run(self):
        for itime in range(self.iStart+1, self.grid.nt+1):
            self.update(itime, final=(itime == self.grid.nt))
        
    
    def movie(self, outfile, fps=1):
        self.plot.nPlot = 1
        
        ani = animation.FuncAnimation(self.plot.figure, self.update, np.arange(1, self.grid.nt+1), 
                                      init_func=self.init, repeat=False, blit=True)
        ani.save(outfile, fps=fps)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vlasov-Poisson Solver in 1D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='Run HDF5 File')
    parser.add_argument('-np', metavar='i', type=int, default=1,
                        help='plot every i\'th frame')    
    parser.add_argument('-ns', metavar='i', type=int, default=0,
                        help='start at frame i')    
    parser.add_argument('-o', metavar='<run.mp4>', type=str, default=None,
                        help='output video file')    
    
    args = parser.parse_args()
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = replay(args.hdf5_file, args.np, args.ns)
    
    print
    input('Hit any key to start replay.')
    print
    
    if args.o != None:
        pyvp.movie(args.o, args.np, args.ns)
    else:
        pyvp.run()
    
    print
    print("Replay finished.")
    print
    
