'''
Created on Mar 21, 2012

@author: mkraus
'''

import matplotlib.pyplot as plt


class PlotBGK(object):
    '''
    classdocs
    '''

    def __init__(self, grid, distribution, hamiltonian, potential, iTime=-1, write=False):
        '''
        Constructor
        '''
        
        self.prefix = '_pyVlasov1D_bgk_'
        
        self.grid         = grid
        self.distribution = distribution
        self.hamiltonian  = hamiltonian
        self.potential    = potential
        
        self.dpi = 100
        
        
        if iTime >= 0 and iTime <= self.grid.nt:
            self.iTime = iTime
        else:
            self.iTime = self.grid.nt
        
        
        self.write = write
        
        self.f = self.distribution.f_ext
        self.h = self.hamiltonian.h_ext
        

        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(14,9), dpi=self.dpi)
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        plt.subplots_adjust(left=0.07, right=0.96, top=0.93, bottom=0.07)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.97, 't = %1.2f' % (grid.t[self.iTime]), horizontalalignment='center') 
        
        # create scatter plot
        plt.scatter(self.h, self.f)
        plt.xlabel('Total Energy')
        plt.ylabel('Distribution Function')
        
        if self.write:
            filename = self.prefix + str('%06d' % (self.iTime)) + '.png'
            plt.savefig(filename, dpi=self.dpi)
        else:
            plt.draw()
            plt.show(block=True)
