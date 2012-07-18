'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

#import matplotlib as mpl

import StringIO
import argparse
import os

import numpy as np
import h5py

import core
#import plot


class movie(object):
    '''
    
    '''


    def __init__(self, hdf5_file, ntMax=0, nTime=0, iStart=0, nPlot=1, vMax=0.0, cMax=False, cFac=1.0, write=False):
        '''
        Constructor
        '''
        
        self.iStart = iStart
        self.nPlot  = nPlot
        
        self.hdf5_files = []
        
        self.hdf5 = h5py.File(hdf5_file, 'r')
        
#        for file in hdf5_files.split(','):
#            print("  Opening %s" % (file))
#            self.hdf5_files.append(h5py.File(file, 'r'))
#        
#        print
        
#         # read config file from HDF5 and create config object
#        cfg_str = self.hdf5_files[0]['runcfg'][:][0]
#        
#        cfg_io = StringIO.StringIO(cfg_str.strip())
#        cfg     = core.Config(cfg_io)
#        cfg_io.close()
        
        self.grid         = core.Grid                (hdf5_in=self.hdf5, replay=True)
        self.potential    = core.Potential           (self.grid, hdf5_in=self.hdf5, replay=True,
                                                      poisson_const=-1.)
        self.hamiltonian  = core.Hamiltonian         (self.grid, hdf5_in=self.hdf5, replay=True)
        self.distribution = core.DistributionFunction(self.grid, hdf5_in=self.hdf5, replay=True)
        
#        for ifile in range(1, len(self.hdf5_files)):
#            self.grid.append_time(self.hdf5_files[ifile]['t'][1:,0,0])
                
        self.potential.read_from_hdf5(iStart)
        self.distribution.read_from_hdf5(iStart)
        self.hamiltonian.read_from_hdf5(iStart)

        
        if ntMax < self.grid.nt:
            self.nt = ntMax
        else:
            self.nt = self.grid.nt
        
        self.plot = PlotMovie(self.grid, self.distribution, self.hamiltonian, self.potential,
                              nTime, nPlot, self.nt, vMax, cMax, cFac, write)
        
    
    def __del__(self):
        self.hdf5.close()
        
#        if self.hdf5_files != None:
#            for hdf5 in self.hdf5_files:
#                hdf5.close()
        
    
    def init(self):
        self.update(0)
    
    
    def update(self, itime, final=False):
        self.potential.read_from_hdf5(itime)
        self.distribution.read_from_hdf5(itime)
        self.hamiltonian.read_from_hdf5(itime)
        
        if itime > 0:
            self.plot.add_timepoint()
        
        return self.plot.update(final=final)
    
    
    def run(self, write=False):
        ttime = 0
        
#        for hdf5 in self.hdf5_files:
#        nt = len(hdf5['grid']['t'][:]) - 1
        
        self.potential.set_hdf5_file(self.hdf5)
        self.hamiltonian.set_hdf5_file(self.hdf5)
        self.distribution.set_hdf5_file(self.hdf5)

        for itime in range(1, self.nt+1):
#            if self.ntMax > 0 and ttime >= self.ntMax:
#                break
#            
#            self.update(itime, final=(itime == nt and self.hdf5 == self.hdf5_files[-1]))
            self.update(itime, final=(itime == self.nt))
            ttime += 1
        
    
    def movie(self, outfile, fps=10):
        self.run(write=True)
        
        command = ('mencoder',
                   'mf://*.png',
                   '-mf',
                   'type=png:w=980:h=630:fps=' + str(fps),
                   '-ovc',
                   'x264',
#                   'lavc',
#                   '-lavcopts',
#                   'vcodec=mpeg4', 
                   '-oac',
                   'copy',
                   '-o',
                   outfile)
        
        os.spawnvp(os.P_WAIT, 'mencoder', command)
        
    

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
    parser.add_argument('-v', metavar='f', type=float, default=0.0,
                        help='limit velocity domain to +/-v')
    parser.add_argument('-cmax', metavar='b', type=bool, default=False,
                        help='use max values of simulation in contour plots')
    parser.add_argument('-cfac', metavar='f', type=float, default=1.0,
                        help='multiply max value of initial data in contour plots')
    parser.add_argument('-o', metavar='<run.mp4>', type=str, default=None,
                        help='output video file')    
    parser.add_argument('-fps', metavar='i', type=int, default=1,
                        help='frames per second')    
    
    args = parser.parse_args()
    
    if args.o != None:
        write = True
        mpl.use('AGG')
    else:
        write = False
    

    from plot.plot_movie import PlotMovie
    
    
    print
    print("Replay run with " + args.hdf5_file)
    print
    
    pyvp = movie(args.hdf5_file, ntMax=args.ntmax, nTime=args.nt, nPlot=args.np,
                 vMax=args.v, cMax=args.cmax, cFac=args.cfac, write=write)
    
    if not write:
        print
        raw_input('Hit any key to start replay.')
        print
    
    if write:
        pyvp.movie(args.o, args.fps)
    else:
        pyvp.run()
    
    print
    print("Replay finished.")
    print
    
