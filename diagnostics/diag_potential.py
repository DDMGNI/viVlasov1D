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
    Plots the time evolution of the potential energy.
    '''


    def __init__(self, hdf5_file, nPlot=1, nmax=0):
        '''
        Constructor
        '''
        
        self.nPlot = nPlot
        
        self.hdf5      = h5py.File(hdf5_file, 'r')
        self.grid      = Grid().load_from_hdf5(self.hdf5)
        self.potential = Potential(self.grid, self.hdf5)
        
        
        self.energy = np.zeros(self.grid.nt+1)
        
        for itime in range(0, self.grid.nt+1):
            self.potential.read_from_hdf5(itime)
            self.energy[itime] = np.sqrt(self.potential.charge * self.potential.E)

        tMax = []
        EMax = []
        
        # find maxima
        for it in range(0, self.grid.nt+1):
            itm = (it-1+self.grid.nt+1) % (self.grid.nt+1)
            itp = (it+1+self.grid.nt+1) % (self.grid.nt+1)
            
            if self.energy[it] > self.energy[itm] and self.energy[it] > self.energy[itp]:
                tMax.append(self.grid.t[it])
                EMax.append(self.energy[it])
        
        # fit maxima
        if nmax == 0:
            nmax = len(tMax)
        else:
            nmax += 1
        
        fit = np.polyfit(tMax[1:nmax], np.log(EMax[1:nmax]), 1)
        fit_fn = np.poly1d(fit)
        
        print("")
        print("Fit Parameter (m,b):", fit)
        print("")
        
        # set up figure/window size
        self.figure1 = plt.figure(num=None, figsize=(16,9))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.09, top=0.94, wspace=0.1, hspace=0.2)
        
        # plot
        plt.semilogy(self.grid.t, self.energy)
        plt.plot(tMax, EMax, 'ro')
        plt.plot(self.grid.t, np.exp(fit_fn(self.grid.t)), '--k')
        
#        plt.title("Damping of the Electrostatic Field", fontsize=24)
        plt.xlabel("$t$", labelpad=15, fontsize=22)
        plt.ylabel("$\parallel E (x,t) \parallel_{2}$", fontsize=22)
        plt.tight_layout()
       
        plt.draw()
        
        
        if hdf5_file.rfind('/') >= 0:
            filename = hdf5_file[hdf5_file.rfind('/'):]
        else:
            filename = hdf5_file
        
        filename  = filename.replace('.hdf5', '_')
        filename += 'potential'
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pdf')
       
    
    def __del__(self):
        if self.hdf5 != None:
            self.hdf5.close()
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vlasov-Poisson Solver in 1D')
    
    parser.add_argument('hdf5_file', metavar='<run.hdf5>', type=str,
                        help='HDF5 data file')
    parser.add_argument('-np', metavar='i', type=int, default=1,
                        help='plot every i\'th frame')
    parser.add_argument('-nmax', metavar='i', type=int, default=1,
                        help='fit only until i\'th maximum')
    
    args = parser.parse_args()
    
    print
    print("Plot field energy decay for run with " + args.hdf5_file)
    print
    
    pot = potential(args.hdf5_file, args.np, args.nmax)
    
    
