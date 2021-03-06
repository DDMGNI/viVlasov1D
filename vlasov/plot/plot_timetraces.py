'''
Created on Mar 21, 2012

@author: mkraus
'''

import numpy as np

from scipy.ndimage     import zoom, gaussian_filter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import ScalarFormatter, MaxNLocator


class PlotTimetraces(object):
    '''
    classdocs
    '''

    def __init__(self, grid, distribution, hamiltonian, potential, first=-1, last=0, vMax=0.0):
        '''
        Constructor
        '''
        
        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', family='sans-serif', size='24')
        
        if last > 0 and last < grid.nt:
            self.nTime = last
        else:
            self.nTime = grid.nt
            
        if first > 0 and first < grid.nt:
            self.iStart = first
        else:
            self.iStart = 0
        
        
        self.iTime  = 0
        self.nPlot  = 1
        self.vMax   = vMax
        
        self.grid         = grid
        self.distribution = distribution
        self.hamiltonian  = hamiltonian
        self.potential    = potential
        
        self.partnum   = np.zeros(grid.nt+1)
        self.energy    = np.zeros(grid.nt+1)
        self.momentum  = np.zeros(grid.nt+1)
        self.entropy   = np.zeros(grid.nt+1)
        self.L1        = np.zeros(grid.nt+1)
        self.L2        = np.zeros(grid.nt+1)
        
        self.x       = np.zeros(grid.nx+1)
        
        self.x[0:-1] = self.grid.x
        self.x[  -1] = self.grid.xLength()
        
        
        # set up tick formatter
        majorFormatter = ScalarFormatter(useOffset=False)
        ## -> limit to 1.1f precision
        majorFormatter.set_powerlimits((-1,+1))
        majorFormatter.set_scientific(True)
        
        
        # add data for zero timepoint
        self.add_timepoint()
        
        # set up plots
        self.axes  = {}
        self.conts = {}
        self.cbars = {}
        self.lines = {}
        
        self.update_boundaries()
        


        # distribution function (filled contour)
        self.figure1 = plt.figure(num=1, figsize=(16,9))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.09, top=0.94, wspace=0.1, hspace=0.2)

        self.axes["f"] = plt.subplot(1,1,1)
#        self.axes["f"].set_title('Distribution Function $f (x,v)$')
        self.axes["f"].title.set_y(1.01)


        self.figure2 = plt.figure(num=2, figsize=(16,10))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.95)

        gs = gridspec.GridSpec(3, 1)
        self.axes["N"] = plt.subplot(gs[0,0])
        self.axes["E"] = plt.subplot(gs[1,0])
        self.axes["P"] = plt.subplot(gs[2,0])
        
        
        self.figure3 = plt.figure(num=3, figsize=(16,4))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.22, top=0.88, wspace=0.10, hspace=0.2)

        self.axes["E0"] = plt.subplot(1,1,1)


        self.figure4 = plt.figure(num=4, figsize=(16,4))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.22, top=0.88, wspace=0.10, hspace=0.2)

        self.axes["P0"] = plt.subplot(1,1,1)


        self.figure5 = plt.figure(num=5, figsize=(16,4))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.22, top=0.88, wspace=0.10, hspace=0.2)

        self.axes["N0"] = plt.subplot(1,1,1)


        self.figure6 = plt.figure(num=6, figsize=(16,4))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.22, top=0.88, wspace=0.10, hspace=0.2)

        self.axes["L1"] = plt.subplot(1,1,1)


        self.figure7 = plt.figure(num=7, figsize=(16,4))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.22, top=0.88, wspace=0.10, hspace=0.2)

        self.axes["L2"] = plt.subplot(1,1,1)


        self.figure8 = plt.figure(num=8, figsize=(16,4))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.22, top=0.88, wspace=0.10, hspace=0.2)

        self.axes["S0"] = plt.subplot(1,1,1)


        xStart = self.grid.t[self.iStart]
        xEnd   = self.grid.t[self.nTime]

        self.lines["N" ], = self.axes["N" ].plot(self.grid.t[self.iStart:self.nTime+1], self.partnum  [self.iStart:self.nTime+1])
        self.lines["E" ], = self.axes["E" ].plot(self.grid.t[self.iStart:self.nTime+1], self.energy   [self.iStart:self.nTime+1])
        self.lines["P" ], = self.axes["P" ].plot(self.grid.t[self.iStart:self.nTime+1], self.momentum [self.iStart:self.nTime+1])
        self.lines["E0"], = self.axes["E0"].plot(self.grid.t[self.iStart:self.nTime+1], self.energy   [self.iStart:self.nTime+1])
        self.lines["P0"], = self.axes["P0"].plot(self.grid.t[self.iStart:self.nTime+1], self.momentum [self.iStart:self.nTime+1])
        self.lines["N0"], = self.axes["N0"].plot(self.grid.t[self.iStart:self.nTime+1], self.partnum  [self.iStart:self.nTime+1])
        self.lines["L1"], = self.axes["L1"].plot(self.grid.t[self.iStart:self.nTime+1], self.L1       [self.iStart:self.nTime+1])
        self.lines["L2"], = self.axes["L2"].plot(self.grid.t[self.iStart:self.nTime+1], self.L2       [self.iStart:self.nTime+1])
        self.lines["S0"], = self.axes["S0"].plot(self.grid.t[self.iStart:self.nTime+1], self.entropy  [self.iStart:self.nTime+1])
        
        self.axes ["N" ].set_title('Total Particle Number Error $\Delta N (t)$', fontsize=24)
        self.axes ["E" ].set_title('Total Energy Error $\Delta E (t)$', fontsize=24)
        self.axes ["E0"].set_title('Total Energy Error $\Delta E (t)$', fontsize=24)
        
        if self.hamiltonian.P0 == 0:
            self.axes ["P" ].set_title('Total Momentum $P (t)$', fontsize=24)
            self.axes ["P0"].set_title('Total Momentum $P (t)$', fontsize=24)
        else:
            self.axes ["P" ].set_title('Total Momentum Error $P (t)$', fontsize=24)
            self.axes ["P0"].set_title('Total Momentum Error $\Delta P (t)$', fontsize=24)
            
        self.axes ["N0"].set_title('Total Particle Number Error $\Delta N (t)$', fontsize=24)
        self.axes ["L1"].set_title('$L_{1}$ Integral Norm Error $\Delta L_{1} (t)$', fontsize=24)
        self.axes ["L2"].set_title('$L_{2}$ Integral Norm Error $\Delta L_{2} (t)$', fontsize=24)
        self.axes ["S0"].set_title('Entropy Error $\Delta S (t)$', fontsize=24)

        self.axes ["N" ].title.set_y(1.02)
        self.axes ["E" ].title.set_y(1.02)
        self.axes ["P" ].title.set_y(1.02)
        self.axes ["E0"].title.set_y(1.02)
        self.axes ["P0"].title.set_y(1.02)
        self.axes ["N0"].title.set_y(1.02)
        self.axes ["L1"].title.set_y(1.00)
        self.axes ["L2"].title.set_y(1.00)
        self.axes ["S0"].title.set_y(1.02)
        
        self.axes ["N" ].set_ylabel('$(N - N_0) / N_0$', fontsize=22)
        self.axes ["E" ].set_ylabel('$(E - E_0) / E_0$', fontsize=22)
        self.axes ["E0"].set_ylabel('$(E - E_0) / E_0$', fontsize=22)
        
        if self.hamiltonian.P0 == 0:
            self.axes ["P" ].set_ylabel('$P$', fontsize=22)
            self.axes ["P0"].set_ylabel('$P$', fontsize=22)
        else:
            self.axes ["P" ].set_ylabel('$(P - P_0) / P_0$', fontsize=22)
            self.axes ["P0"].set_ylabel('$(P - P_0) / P_0$', fontsize=22)

        self.axes ["N0"].set_ylabel('$(N - N_0) / N_0$', fontsize=22)
        self.axes ["L1"].set_ylabel('$(L_1 - L_{1,0}) / L_{1,0}$', fontsize=22)
        self.axes ["L2"].set_ylabel('$(L_2 - L_{2,0}) / L_{2,0}$', fontsize=22)
        self.axes ["S0"].set_ylabel('$(S - S_0) / S_0$', fontsize=22)
        
        self.axes ["N" ].yaxis.set_label_coords(-0.07, 0.5)
        self.axes ["E" ].yaxis.set_label_coords(-0.07, 0.5)
        self.axes ["P" ].yaxis.set_label_coords(-0.07, 0.5)
        self.axes ["E0"].yaxis.set_label_coords(-0.07, 0.5)
        self.axes ["P0"].yaxis.set_label_coords(-0.07, 0.5)
        self.axes ["N0"].yaxis.set_label_coords(-0.07, 0.5)
        self.axes ["L1"].yaxis.set_label_coords(-0.07, 0.5)
        self.axes ["L2"].yaxis.set_label_coords(-0.07, 0.5)
        self.axes ["S0"].yaxis.set_label_coords(-0.07, 0.5)

        self.axes ["N" ].set_xlim((xStart,xEnd))
        self.axes ["E" ].set_xlim((xStart,xEnd))
        self.axes ["P" ].set_xlim((xStart,xEnd))
        self.axes ["E0"].set_xlim((xStart,xEnd))
        self.axes ["P0"].set_xlim((xStart,xEnd))
        self.axes ["N0"].set_xlim((xStart,xEnd))
        self.axes ["L1"].set_xlim((xStart,xEnd))
        self.axes ["L2"].set_xlim((xStart,xEnd))
        self.axes ["S0"].set_xlim((xStart,xEnd))
        
        self.axes ["N" ].yaxis.set_major_formatter(majorFormatter)
        self.axes ["E" ].yaxis.set_major_formatter(majorFormatter)
        self.axes ["P" ].yaxis.set_major_formatter(majorFormatter)
        self.axes ["E0"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["P0"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["N0"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["L1"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["L2"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["S0"].yaxis.set_major_formatter(majorFormatter)
        
        self.axes ["N" ].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["E" ].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["P" ].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["E0"].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["P0"].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["N0"].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["L1"].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["L2"].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["S0"].yaxis.set_major_locator(MaxNLocator(4))
        
        self.axes ["f" ].set_xlabel('$x$', labelpad=15)
        self.axes ["f" ].set_ylabel('$v$', labelpad=15)
        self.axes ["P" ].set_xlabel('$t$', labelpad=15)
        self.axes ["E0"].set_xlabel('$t$', labelpad=15)
        self.axes ["P0"].set_xlabel('$t$', labelpad=15)
        self.axes ["N0"].set_xlabel('$t$', labelpad=15)
        self.axes ["L1"].set_xlabel('$t$', labelpad=15)
        self.axes ["L2"].set_xlabel('$t$', labelpad=15)
        self.axes ["S0"].set_xlabel('$t$', labelpad=15)
        
        for ax in self.axes:
            for tick in self.axes[ax].xaxis.get_major_ticks():
                tick.set_pad(12)
            for tick in self.axes[ax].yaxis.get_major_ticks():
                tick.set_pad(8)
        
        # switch off some ticks
        plt.setp(self.axes["N"].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"].get_xticklabels(), visible=False)
        
        self.figure2 = plt.figure(num=2, figsize=(16,10))
        plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.4)
        
        self.update()
        
        
        
    def save_plots(self):
        
        plt.figure(1)
        filename = str('F_%06d' % (self.iTime-1))
        plt.savefig(filename + '.png', dpi=300)
        
        plt.figure(2)
        filename = str('NEP_%06d' % (self.iTime-1))
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pdf')

        plt.figure(3)
        filename = str('E_%06d' % (self.iTime-1))
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pdf')

        plt.figure(4)
        filename = str('P_%06d' % (self.iTime-1))
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pdf')

        plt.figure(5)
        filename = str('N_%06d' % (self.iTime-1))
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pdf')

        plt.figure(6)
        filename = str('L1_%06d' % (self.iTime-1))
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pdf')

        plt.figure(7)
        filename = str('L2_%06d' % (self.iTime-1))
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pdf')

        plt.figure(8)
        filename = str('S_%06d' % (self.iTime-1))
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pdf')


    
    def update_boundaries(self):
        self.fmin = +1e40
        self.fmax = -1e40
        
        self.fmin = min(self.fmin, self.distribution.f.min() )
        self.fmax = max(self.fmax, self.distribution.f.max() )

        self.fmin += 0.2 * (self.fmax-self.fmin)

        self.fnorm  = colors.Normalize(vmin=self.fmin, vmax=self.fmax)
        
        
    
    def update(self, final=False):
        
        if not (self.iTime == 1 or (self.iTime-1) % self.nPlot == 0 or self.iTime-1 == self.nTime):
            return
        
        for ckey, cont in self.conts.items():
            for coll in cont.collections:
                self.axes[ckey].collections.remove(coll)
        
#        nscale = 5
#        fint = zoom(self.distribution.f.T, nscale)
#        xint = np.linspace(self.grid.x[0], self.grid.x[-1], nscale*len(self.grid.x))
#        vint = np.linspace(self.grid.v[0], self.grid.v[-1], nscale*len(self.grid.v))
#
#        self.conts["f"] = self.axes["f"].contourf(xint, vint, fint, 100, norm=self.fnorm, extend='neither')

#         fint = gaussian_filter(self.distribution.f.T, sigma=1.0, order=0)

        self.conts["f"] = self.axes["f"].contourf(self.grid.x, self.grid.v, self.distribution.f.T, 100, norm=self.fnorm, extend='neither')

#        self.conts["f"] = self.axes["f"].contourf(self.grid.x, self.grid.v, fint, 100, norm=self.fnorm, extend='neither')
        self.axes ["f"].set_title('t = %.0f' % (self.grid.t[self.iTime-1]))
        
        if self.vMax > 0.0:
            self.axes["f"].set_ylim((-self.vMax, +self.vMax)) 
        
        
        xStart = self.grid.t[self.iStart]
        xEnd   = self.grid.t[self.nTime]
        
        self.lines["N"].set_xdata(self.grid.t[self.iStart:self.nTime+1])
        self.lines["N"].set_ydata(self.partnum[self.iStart:self.nTime+1])
        self.axes ["N"].relim()
        self.axes ["N"].autoscale_view()
        self.axes ["N"].set_xlim((xStart,xEnd)) 
        
        self.lines["E"].set_xdata(self.grid.t[self.iStart:self.nTime+1])
        self.lines["E"].set_ydata(self.energy[self.iStart:self.nTime+1])
        self.axes ["E"].relim()
        self.axes ["E"].autoscale_view()
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        
        self.lines["P"].set_xdata(self.grid.t[self.iStart:self.nTime+1])
        self.lines["P"].set_ydata(self.momentum[self.iStart:self.nTime+1])
        self.axes ["P"].relim()
        self.axes ["P"].autoscale_view()
        self.axes ["P"].set_xlim((xStart,xEnd)) 
        
        self.lines["E0"].set_xdata(self.grid.t[self.iStart:self.nTime+1])
        self.lines["E0"].set_ydata(self.energy[self.iStart:self.nTime+1])
        self.axes ["E0"].relim()
        self.axes ["E0"].autoscale_view()
        self.axes ["E0"].set_xlim((xStart,xEnd)) 
        
        self.lines["P0"].set_xdata(self.grid.t[self.iStart:self.nTime+1])
        self.lines["P0"].set_ydata(self.momentum[self.iStart:self.nTime+1])
        self.axes ["P0"].relim()
        self.axes ["P0"].autoscale_view()
        self.axes ["P0"].set_xlim((xStart,xEnd)) 
        
        self.lines["N0"].set_xdata(self.grid.t[self.iStart:self.nTime+1])
        self.lines["N0"].set_ydata(self.partnum[self.iStart:self.nTime+1])
        self.axes ["N0"].relim()
        self.axes ["N0"].autoscale_view()
        self.axes ["N0"].set_xlim((xStart,xEnd)) 
        
        self.lines["L1"].set_xdata(self.grid.t[self.iStart:self.nTime+1])
        self.lines["L1"].set_ydata(self.L1[self.iStart:self.nTime+1])
        self.axes ["L1"].relim()
        self.axes ["L1"].autoscale_view()
        self.axes ["L1"].set_xlim((xStart,xEnd)) 
        
        self.lines["L2"].set_xdata(self.grid.t[self.iStart:self.nTime+1])
        self.lines["L2"].set_ydata(self.L2[self.iStart:self.nTime+1])
        self.axes ["L2"].relim()
        self.axes ["L2"].autoscale_view()
        self.axes ["L2"].set_xlim((xStart,xEnd)) 
        
        self.lines["S0"].set_xdata(self.grid.t[self.iStart:self.nTime+1])
        self.lines["S0"].set_ydata(self.entropy[self.iStart:self.nTime+1])
        self.axes ["S0"].relim()
        self.axes ["S0"].autoscale_view()
        self.axes ["S0"].set_xlim((xStart,xEnd)) 
        
        
        for i in range(0,8):
            plt.figure(i+1)
            plt.draw()
        
    
    def add_timepoint(self):
#         E0 = self.hamiltonian.E0
#         E  = self.hamiltonian.E
        
        E  = self.hamiltonian.E_kin  + self.hamiltonian.E_pot  + self.potential.E
        E0 = self.hamiltonian.E_kin0 + self.hamiltonian.E_pot0 + self.potential.E0
        
        E_error   = (E - E0) / E0
        
        self.energy   [self.iTime] = E_error
        
        if self.hamiltonian.P0 == 0:
            self.momentum[self.iTime] = self.hamiltonian.P
        else:
            self.momentum[self.iTime] = self.hamiltonian.P_error
        
        self.partnum  [self.iTime] = self.distribution.N_error
        self.entropy  [self.iTime] = self.distribution.S_error
        self.L1       [self.iTime] = self.distribution.L1_error
        self.L2       [self.iTime] = self.distribution.L2_error
        
        self.iTime += 1
