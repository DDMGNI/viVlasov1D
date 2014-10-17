'''
Created on Mar 21, 2012

@author: mkraus
'''

import numpy as np

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
        plt.scatter(self.h, self.f, marker='o', c='b', s=1)
        plt.xlabel('Total Energy')
        plt.ylabel('Distribution Function')
        plt.xlim((-3,+12))
#        plt.xlim((-5,+35))
        
        
        print()
        print("BGK Analysis")
        print("============")
        print()
        
        fmax = self.distribution.f.max()
        hmin = self.hamiltonian.h.min()
        
        fi = 0
        fj = 0
        hi = 0
        hj = 0

        r1i = 0
        r1j = 0
        r2i = 0
        r2j = 0
        
        for i in range(self.grid.nx):
            for j in range(self.grid.nv):
                if self.distribution.f[i,j] < 0.27 and self.distribution.f[i,j] > 0.22 and self.hamiltonian.h[i,j] < self.hamiltonian.h[fi,fj]:
                    fi = i
                    fj = j
                
                if self.hamiltonian.h[i,j] == hmin:
                    hi = i
                    hj = j
        
        for i in range(self.grid.nx):
            for j in range(self.grid.nv):
                if self.distribution.f[i,j] < 0.27 and self.distribution.f[i,j] > 0.22 and self.hamiltonian.h[i,j] > self.hamiltonian.h[fi,fj]:
                    r1i = i
                    r1j = j
        
                if self.distribution.f[i,j] < 0.17 and self.distribution.f[i,j] > 0.12 and self.hamiltonian.h[i,j] > self.hamiltonian.h[fi,fj]:
                    r2i = i
                    r2j = j
        
        
        slope1 = (self.distribution.f[fi,fj] - self.distribution.f[hi,hj]) \
               / (self.hamiltonian.h[fi,fj]  - self.hamiltonian.h[hi,hj])
        
        slope2 = (self.distribution.f[r1i,r1j] - self.distribution.f[r2i,r2j]) \
               / (self.hamiltonian.h[r1i,r1j]  - self.hamiltonian.h[r2i,r2j])
        
        print("min(phi):  %16.8E" % np.min(self.hamiltonian.h1))
        print("max(phi):  %16.8E" % np.max(self.hamiltonian.h1))
        print()
        print("min(h):    %16.8E" % hmin)
        print("x(min(h)): %16.8E" % self.grid.x[hi])
        print("v(min(h)): %16.8E" % self.grid.v[hj])
        print()
        print("max(f):    %16.8E" % fmax)
        print("x(max(f)): %16.8E" % self.grid.x[fi])
        print("v(max(f)): %16.8E" % self.grid.v[fj])
        print("h(max(f)): %16.8E" % self.hamiltonian.h[fi,fj])
        print()
        print("lslope:    %16.8E" % slope1)
        print("lslope^-1: %16.8E" % (1./slope1))
        print()
        print("rslope:    %16.8E" % slope2)
        print("rslope^-1: %16.8E" % (1./slope2))
        print()
        
        print()
        print()
        
        
        plt.plot([self.hamiltonian.h[hi,hj], self.hamiltonian.h[fi,fj] + 0.2], [self.distribution.f[hi,hj], self.distribution.f[fi,fj] + 0.2 * slope1], color='r')
        plt.plot([self.hamiltonian.h[r1i,r1j] - 1., self.hamiltonian.h[r2i,r2j] + 1.5], [self.distribution.f[r1i,r1j] - 1.0 * slope2, self.distribution.f[r2i,r2j] + 1.5 * slope2], color='g')
        
        plt.figtext(0.1, 0.6, "$s_{l}^{-1} = %16.8E$" % (1./slope1))
        plt.figtext(0.4, 0.6, "$s_{r}^{-1} = %16.8E$" % (1./slope2))
        
        if self.write:
            filename = self.prefix + str('%06d' % (self.iTime)) + '.png'
            plt.savefig(filename, dpi=self.dpi)
        else:
            plt.draw()
            plt.show(block=True)
