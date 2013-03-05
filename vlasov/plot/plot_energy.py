'''
Created on Mar 21, 2012

@author: mkraus
'''

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import ScalarFormatter, MaxNLocator


class PlotEnergy(object):
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
        
        self.partnum   = np.zeros_like(grid.tGrid)
        self.enstrophy = np.zeros_like(grid.tGrid)
        self.energy    = np.zeros_like(grid.tGrid)
        self.momentum  = np.zeros_like(grid.tGrid)
        
        self.x       = np.zeros(grid.nx+1)
        
        self.x[0:-1] = self.grid.xGrid
        self.x[  -1] = self.grid.L
        
        
        # set up tick formatter
        majorFormatter = ScalarFormatter(useOffset=False)
        ## -> limit to 1.1f precision
        majorFormatter.set_powerlimits((-1,+1))
        majorFormatter.set_scientific(True)
        
        
        # set up plot margins
#        plt.subplots_adjust(hspace=0.2, wspace=0.25)
#        plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.05)
        
        # set up plot title
#        self.title = self.figure.text(0.5, 0.97, 't = 0.0' % (self.grid.tGrid[self.iTime]), horizontalalignment='center') 
        
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
        self.axes ["f"].set_title('Distribution Function $f (x,v)$')
        self.axes ["f"].title.set_y(1.01)
#        self.conts["f"] = self.axes["f"].contourf(self.grid.xGrid, self.grid.vGrid, self.distribution.f.T,  100, norm=self.fnorm, extend='neither')
#        self.cbars["f"] = plt.colorbar(self.conts["f"], orientation='vertical')


        self.figure2 = plt.figure(num=2, figsize=(16,10))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.95)

        gs = gridspec.GridSpec(4, 1)
        self.axes["N"] = plt.subplot(gs[0,0])
        self.axes["L"] = plt.subplot(gs[1,0])
        self.axes["E"] = plt.subplot(gs[2,0])
        self.axes["P"] = plt.subplot(gs[3,0])
        
        
        self.figure3 = plt.figure(num=3, figsize=(16,4))
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.22, top=0.88, wspace=0.10, hspace=0.2)

        self.axes["E0"] = plt.subplot(1,1,1)


        tStart, tEnd, xStart, xEnd = self.get_timerange()

        self.lines["N" ], = self.axes["N" ].plot(self.grid.tGrid[tStart:tEnd], self.partnum  [tStart:tEnd])
        self.lines["L" ], = self.axes["L" ].plot(self.grid.tGrid[tStart:tEnd], self.enstrophy[tStart:tEnd])
        self.lines["E" ], = self.axes["E" ].plot(self.grid.tGrid[tStart:tEnd], self.energy   [tStart:tEnd])
        self.lines["P" ], = self.axes["P" ].plot(self.grid.tGrid[tStart:tEnd], self.momentum [tStart:tEnd])
        self.lines["E0"], = self.axes["E0"].plot(self.grid.tGrid[tStart:tEnd], self.energy   [tStart:tEnd])
        
        self.axes ["N" ].set_title('Total Particle Number Error $\Delta N (t)$', fontsize=24)
        self.axes ["L" ].set_title('$L_{2}$ Integral Norm Error $\Delta L_{2} (t)$', fontsize=24)
        self.axes ["E" ].set_title('Total Energy Error $\Delta E (t)$', fontsize=24)
        self.axes ["P" ].set_title('Total Momentum $P (t)$', fontsize=24)
        self.axes ["E0"].set_title('Total Energy Error $\Delta E (t)$', fontsize=24)

        self.axes ["N"].title.set_y(1.02)
        self.axes ["L"].title.set_y(1.00)
        self.axes ["E"].title.set_y(1.02)
        self.axes ["P"].title.set_y(1.02)
        self.axes ["E0"].title.set_y(1.02)
        
        self.axes ["N" ].set_ylabel('$(N - N_0) / N_0$', fontsize=20)
        self.axes ["L" ].set_ylabel('$(L_2 - L_{2,0}) / L_{2,0}$', fontsize=20)
        self.axes ["E" ].set_ylabel('$(E - E_0) / E_0$', fontsize=20)
        self.axes ["P" ].set_ylabel('$(P - P_0) / P_0$', fontsize=20)
        self.axes ["E0"].set_ylabel('$(E - E_0) / E_0$')
        
        self.axes ["N"].yaxis.set_label_coords(-0.08, 0.5)
        self.axes ["L"].yaxis.set_label_coords(-0.08, 0.5)
        self.axes ["E"].yaxis.set_label_coords(-0.08, 0.5)
        self.axes ["P"].yaxis.set_label_coords(-0.08, 0.5)

        self.axes ["N" ].set_xlim((xStart,xEnd)) 
        self.axes ["L" ].set_xlim((xStart,xEnd)) 
        self.axes ["E" ].set_xlim((xStart,xEnd)) 
        self.axes ["P" ].set_xlim((xStart,xEnd)) 
        self.axes ["E0"].set_xlim((xStart,xEnd)) 
        
        self.axes ["N" ].yaxis.set_major_formatter(majorFormatter)
        self.axes ["L" ].yaxis.set_major_formatter(majorFormatter)
        self.axes ["E" ].yaxis.set_major_formatter(majorFormatter)
        self.axes ["P" ].yaxis.set_major_formatter(majorFormatter)
        self.axes ["E0"].yaxis.set_major_formatter(majorFormatter)
        
        self.axes ["N" ].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["L" ].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["E" ].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["P" ].yaxis.set_major_locator(MaxNLocator(4))
        self.axes ["E0"].yaxis.set_major_locator(MaxNLocator(4))
        
        self.axes ["f" ].set_xlabel('$x$', labelpad=15)
        self.axes ["f" ].set_ylabel('$v$', labelpad=15)
        self.axes ["P" ].set_xlabel('$t$', labelpad=15)
        self.axes ["E0"].set_xlabel('$t$', labelpad=15)
        
        for ax in self.axes:
            for tick in self.axes[ax].xaxis.get_major_ticks():
                tick.set_pad(12)
            for tick in self.axes[ax].yaxis.get_major_ticks():
                tick.set_pad(8)
        
        # switch off some ticks
        plt.setp(self.axes["N"].get_xticklabels(), visible=False)
        plt.setp(self.axes["L"].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"].get_xticklabels(), visible=False)
        
        self.figure2 = plt.figure(num=2, figsize=(16,10))
        plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.4)
        
        self.update()
        
        
        
    def save_plots(self):
        
        plt.figure(1)
        filename = str('F_%06d' % (self.iTime-1))
        plt.savefig(filename + '.png', dpi=300)
#        plt.savefig(filename + '.pdf')
        
        plt.figure(2)
        filename = str('NLEP_%06d' % (self.iTime-1))
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.pdf')

        plt.figure(3)
        filename = str('E_%06d' % (self.iTime-1))
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
        
        self.conts["f"] = self.axes["f"].contourf(self.grid.xGrid, self.grid.vGrid, self.distribution.f.T,  100, norm=self.fnorm, extend='neither')
        
        if self.vMax > 0.0:
            self.axes["f"].set_ylim((-self.vMax, +self.vMax)) 
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()
        
        self.lines["N"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["N"].set_ydata(self.partnum[tStart:tEnd])
        self.axes ["N"].relim()
        self.axes ["N"].autoscale_view()
        self.axes ["N"].set_xlim((xStart,xEnd)) 
        
        self.lines["L"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["L"].set_ydata(self.enstrophy[tStart:tEnd])
        self.axes ["L"].relim()
        self.axes ["L"].autoscale_view()
        self.axes ["L"].set_xlim((xStart,xEnd)) 
        
        self.lines["E"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["E"].set_ydata(self.energy[tStart:tEnd])
        self.axes ["E"].relim()
        self.axes ["E"].autoscale_view()
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        
        self.lines["P"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["P"].set_ydata(self.energy[tStart:tEnd])
        self.axes ["P"].relim()
        self.axes ["P"].autoscale_view()
        self.axes ["P"].set_xlim((xStart,xEnd)) 
        
        self.lines["E0"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["E0"].set_ydata(self.energy[tStart:tEnd])
        self.axes ["E0"].relim()
        self.axes ["E0"].autoscale_view()
        self.axes ["E0"].set_xlim((xStart,xEnd)) 
        
        plt.figure(1)
        plt.draw()
        
        plt.figure(2)
        plt.draw()
        
        plt.figure(3)
        plt.draw()
        
        
#        plt.show(block=final)
        
    
    def add_timepoint(self):
        E0 = self.hamiltonian.E0
        E  = self.hamiltonian.E
        
        E_error   = (E - E0) / E0
        
        self.energy   [self.iTime] = E_error
        self.momentum [self.iTime] = self.hamiltonian.P
        self.partnum  [self.iTime] = self.distribution.N_error
        self.enstrophy[self.iTime] = self.distribution.L2_error
        
#        self.title.set_text('t = %1.2f' % (self.grid.tGrid[self.iTime]))
        
        self.iTime += 1
        
    
    def get_timerange(self):
        tStart = self.iTime
        tEnd   = self.iTime + self.nTime
        
        if tStart < self.iStart:
            tStart = self.iStart
        
        if tEnd > self.nTime:
            tEnd = self.nTime
        
        xStart = self.grid.tGrid[tStart]
        xEnd   = self.grid.tGrid[tEnd]
        
        return tStart, tEnd, xStart, xEnd
    
