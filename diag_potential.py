'''
Created on Apr 06, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse
import numpy as np
import h5py

import matplotlib
matplotlib.use('AGG')
#matplotlib.use('PDF')

import matplotlib.pyplot as plt

from vlasov.core import Grid, Potential


class potential(object):
    '''
    
    '''


    def __init__(self, hdf5_file, nPlot=1, fmax=0):
        '''
        Constructor
        '''
        
        self.nPlot = nPlot
        
        self.hdf5 = h5py.File(hdf5_file, 'r')
        
        self.grid         = Grid                (hdf5_in=self.hdf5, replay=True)
        self.potential    = Potential           (self.grid, hdf5_in=self.hdf5, replay=True,
                                                 poisson_const=-1.)
        
        
        self.energy = np.zeros(self.grid.nt+1)
        
        for itime in range(0, self.grid.nt+1):
            self.potential.read_from_hdf5(itime)
            self.energy[itime] = np.sqrt(self.potential.E)

        tMax = []
        EMax = []
        
        # find maxima
        for it in range(0, self.grid.nt+1):
            itm = (it-1+self.grid.nt+1) % (self.grid.nt+1)
            itp = (it+1+self.grid.nt+1) % (self.grid.nt+1)
            
            if self.energy[it] > self.energy[itm] and self.energy[it] > self.energy[itp]:
                tMax.append(self.grid.tGrid[it])
                EMax.append(self.energy[it])
        
        # fit maxima
        if fmax == 0:
            fmax = len(tMax)
        else:
            fmax += 1
        
        fit = np.polyfit(tMax[1:fmax], np.log(EMax[1:fmax]), 1)
        fit_fn = np.poly1d(fit)
        
        print("Fit Parameter (m,b):", fit)
        
        # set up figure/window size
        self.figure1 = plt.figure(num=None, figsize=(16,9))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.09, top=0.94, wspace=0.1, hspace=0.2)
        
        # plot
        plt.semilogy(self.grid.tGrid, self.energy)
        plt.plot(tMax, EMax, 'ro')
        plt.plot(self.grid.tGrid, np.exp(fit_fn(self.grid.tGrid)), '--k')
        
#        plt.title("Damping of the Electrostatic Field", fontsize=24)
        plt.xlabel("$t$", labelpad=15, fontsize=22)
        plt.ylabel("$\parallel E (x,t) \parallel_{2}$", fontsize=22)
        plt.tight_layout()
       
        plt.draw()
        
        filename = str('potential')
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pdf')
       
    
    def __del__(self):
        if self.hdf5 != None:
            self.hdf5.close()
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vlasov-Poisson Solver in 1D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='Run HDF5 File')
    parser.add_argument('-np', metavar='i', type=int, default=1,
                        help='plot every i\'th frame')
    parser.add_argument('-fmax', metavar='i', type=int, default=1,
                        help='fit only until i\'th maximum')
    
    args = parser.parse_args()
    
    print
    print("Plot Field Decay for run with " + args.hdf5_file)
    print
    
    pot = potential(args.hdf5_file, args.np, args.fmax)
    
    
