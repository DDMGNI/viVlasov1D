'''
Created on Mar 21, 2012

@author: mkraus
'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter


class PlotNoether(object):
    '''
    classdocs
    '''

    def __init__(self, grid, distribution, hamiltonian, potential,nTime=0,  iStart=0, nPlot=1):
        '''
        Constructor
        '''
        
        self.eps = 1E-3
        
        
        if nTime > 0 and nTime < grid.nt:
            self.nTime = nTime
        else:
            self.nTime = grid.nt
            
        self.nTime -= iStart
        self.iTime  = iStart
        self.iStart = iStart
        self.nPlot  = nPlot
        
        self.grid         = grid
        self.distribution = distribution
        self.hamiltonian  = hamiltonian
        self.potential    = potential
        
        self.partnum   = np.zeros_like(grid.tGrid)
        self.momentum  = np.zeros_like(grid.tGrid)
        self.energy    = np.zeros_like(grid.tGrid)
        self.entropy   = np.zeros_like(grid.tGrid)
        
        self.L2        = np.zeros_like(grid.tGrid)
        self.L3        = np.zeros_like(grid.tGrid)
        self.L4        = np.zeros_like(grid.tGrid)
        self.L5        = np.zeros_like(grid.tGrid)
        self.L6        = np.zeros_like(grid.tGrid)
        self.L8        = np.zeros_like(grid.tGrid)
        
        self.fmin      = np.zeros_like(grid.tGrid)
        self.fmax      = np.zeros_like(grid.tGrid)
        
        self.x       = np.zeros(grid.nx+1)
        self.n       = np.zeros(grid.nx+1)
        self.phi     = np.zeros(grid.nx+1)
        
        self.x[0:-1] = self.grid.xGrid
        self.x[  -1] = self.grid.L
        
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.2, wspace=0.26)
        plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.05)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.97, 't = 0.0' % (self.grid.tGrid[self.iTime]), horizontalalignment='center') 
        
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
        
        
        # create subplots
        gs = gridspec.GridSpec(4, 5)
        
        self.axes["f"]  = plt.subplot(gs[0:2,0:2])
        self.axes["h"]  = plt.subplot(gs[2:4,0:2])
        
        self.axes["N"]  = plt.subplot(gs[0,2])
        self.axes["P"]  = plt.subplot(gs[1,2])
        self.axes["E"]  = plt.subplot(gs[2,2])
        self.axes["S"]  = plt.subplot(gs[3,2])
        
        self.axes["L2"] = plt.subplot(gs[0,3])
        self.axes["L3"] = plt.subplot(gs[1,3])
        self.axes["L4"] = plt.subplot(gs[2,3])
        self.axes["L5"] = plt.subplot(gs[3,3])
        
        self.axes["L6"] = plt.subplot(gs[0,4])
        self.axes["L8"] = plt.subplot(gs[1,4])
        
        self.axes["fmin"] = plt.subplot(gs[2,4])
        self.axes["fmax"] = plt.subplot(gs[3,4])
        
        
        # distribution function (filled contour)
        self.axes["f"] = plt.subplot(gs[0:2,0:2])
        self.axes ["f"].set_title('$f (x,v)$')
        self.conts["f"] = self.axes["f"].contourf(self.grid.xGrid, self.grid.vGrid, self.distribution.f.T, 10, norm=self.fnorm)
        self.cbars["f"] = plt.colorbar(self.conts["f"], orientation='vertical')
        
        # Hamilton function (filled contour)
        self.axes["h"] = plt.subplot(gs[2:4,0:2])
        self.axes["h"].set_title('$H (x,v)$')
        self.conts["h"] = self.axes["h"].contourf(self.grid.xGrid, self.grid.vGrid, self.hamiltonian.h.T,  10, norm=self.hnorm)
        self.cbars["h"] = plt.colorbar(self.conts["h"], orientation='vertical')

        
        tStart, tEnd, xStart, xEnd = self.get_timerange()

        self.lines["N"], = self.axes["N"].plot(self.grid.tGrid[tStart:tEnd], self.partnum [tStart:tEnd])
        self.lines["P"], = self.axes["P"].plot(self.grid.tGrid[tStart:tEnd], self.momentum[tStart:tEnd])
        self.lines["E"], = self.axes["E"].plot(self.grid.tGrid[tStart:tEnd], self.energy  [tStart:tEnd])
        self.lines["S"], = self.axes["S"].plot(self.grid.tGrid[tStart:tEnd], self.entropy [tStart:tEnd])
        
        self.axes ["N"].set_title('$\Delta N (t)$')
        
        if np.abs(self.hamiltonian.P0) > self.eps:
            self.axes ["P"].set_title('$\Delta P (t)$')
        else:
            self.axes ["P"].set_title('$P (t)$')
            
        self.axes ["E"].set_title('$\Delta E (t)$')
        self.axes ["S"].set_title('$\Delta S (t)$')

        self.axes ["N"].set_xlim((xStart,xEnd)) 
        self.axes ["P"].set_xlim((xStart,xEnd)) 
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        self.axes ["S"].set_xlim((xStart,xEnd)) 
        
        self.axes ["N"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["P"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["E"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["S"].yaxis.set_major_formatter(majorFormatter)
        
        
        self.lines["L2"], = self.axes["L2"].plot(self.grid.tGrid[tStart:tEnd], self.L2[tStart:tEnd])
        self.lines["L3"], = self.axes["L3"].plot(self.grid.tGrid[tStart:tEnd], self.L3[tStart:tEnd])
        self.lines["L4"], = self.axes["L4"].plot(self.grid.tGrid[tStart:tEnd], self.L4[tStart:tEnd])
        self.lines["L5"], = self.axes["L5"].plot(self.grid.tGrid[tStart:tEnd], self.L5[tStart:tEnd])
        self.lines["L6"], = self.axes["L6"].plot(self.grid.tGrid[tStart:tEnd], self.L6[tStart:tEnd])
        self.lines["L8"], = self.axes["L8"].plot(self.grid.tGrid[tStart:tEnd], self.L8[tStart:tEnd])
        
        self.lines["fmin"], = self.axes["fmin"].plot(self.grid.tGrid[tStart:tEnd], self.fmin[tStart:tEnd])
        self.lines["fmax"], = self.axes["fmax"].plot(self.grid.tGrid[tStart:tEnd], self.fmax[tStart:tEnd])
        
        self.axes ["L2"].set_title('$\Delta L^{2} (t)$')
        self.axes ["L3"].set_title('$\Delta L^{3} (t)$')
        self.axes ["L4"].set_title('$\Delta L^{4} (t)$')
        self.axes ["L5"].set_title('$\Delta L^{5} (t)$')
        self.axes ["L6"].set_title('$\Delta L^{6} (t)$')
        self.axes ["L8"].set_title('$\Delta L^{8} (t)$')

        self.axes ["fmin"].set_title('$f_{min} (t)$')
        self.axes ["fmax"].set_title('$\Delta f_{max} (t)$')
        
        self.axes ["L2"].set_xlim((xStart,xEnd)) 
        self.axes ["L3"].set_xlim((xStart,xEnd)) 
        self.axes ["L4"].set_xlim((xStart,xEnd)) 
        self.axes ["L5"].set_xlim((xStart,xEnd)) 
        self.axes ["L6"].set_xlim((xStart,xEnd)) 
        self.axes ["L8"].set_xlim((xStart,xEnd)) 
        
        self.axes ["fmin"].set_xlim((xStart,xEnd)) 
        self.axes ["fmax"].set_xlim((xStart,xEnd)) 
        
        self.axes ["L2"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["L3"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["L4"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["L5"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["L6"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["L8"].yaxis.set_major_formatter(majorFormatter)
        
        self.axes ["fmin"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["fmax"].yaxis.set_major_formatter(majorFormatter)
        

        # switch off some ticks
        plt.setp(self.axes["f"].get_xticklabels(), visible=False)
        plt.setp(self.axes["N"].get_xticklabels(), visible=False)
        plt.setp(self.axes["P"].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"].get_xticklabels(), visible=False)
        plt.setp(self.axes["L2"].get_xticklabels(), visible=False)
        plt.setp(self.axes["L3"].get_xticklabels(), visible=False)
        plt.setp(self.axes["L4"].get_xticklabels(), visible=False)
        plt.setp(self.axes["L6"].get_xticklabels(), visible=False)
        plt.setp(self.axes["L8"].get_xticklabels(), visible=False)
        plt.setp(self.axes["fmin"].get_xticklabels(), visible=False)
        
        
        self.update()
        
        
        
    def update_boundaries(self):
        Fmin = min(+1e40, self.distribution.fmin )
        Fmax = max(-1e40, self.distribution.fmax )

        Hmin = min(+1e40, self.hamiltonian.h.min() )
        Hmax = max(-1e40, self.hamiltonian.h.max() )


        self.fnorm = colors.Normalize(vmin=Fmin + 0.01 * Fmax, vmax=1.01*Fmax)
#         self.fnorm = colors.Normalize(vmin=Fmin + 0.1 * Fmax, vmax=1.1*Fmax)
#         self.fnorm = colors.Normalize(vmin=Fmin, vmax=1.1*Fmax)
        self.hnorm = colors.Normalize(vmin=Hmin, vmax=Hmax)
        
    
    def update(self, final=False):
        
        if not (self.iTime == 1 or (self.iTime-1) % self.nPlot == 0 or self.iTime-1 == self.nTime):
            return
        
#        self.update_boundaries()

        for ckey, cont in self.conts.items():
            for coll in cont.collections:
                self.axes[ckey].collections.remove(coll)
        
#        self.fnorm = colors.Normalize(vmin=self.fmin, vmax=self.fmax)
        
        self.conts["f"] = self.axes["f"].contourf(self.grid.xGrid, self.grid.vGrid, self.distribution.f.T, 10, norm=self.fnorm)
        self.conts["h"] = self.axes["h"].contourf(self.grid.xGrid, self.grid.vGrid, self.hamiltonian.h.T,  10, norm=self.hnorm)
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()
        
        self.lines["N"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["N"].set_ydata(self.partnum[tStart:tEnd])
        self.axes ["N"].relim()
        self.axes ["N"].autoscale_view()
        self.axes ["N"].set_xlim((xStart,xEnd)) 
        
        self.lines["P"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["P"].set_ydata(self.momentum[tStart:tEnd])
        self.axes ["P"].relim()
        self.axes ["P"].autoscale_view()
        self.axes ["P"].set_xlim((xStart,xEnd)) 
        
        self.lines["E"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["E"].set_ydata(self.energy[tStart:tEnd])
        self.axes ["E"].relim()
        self.axes ["E"].autoscale_view()
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        
        self.lines["S"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["S"].set_ydata(self.entropy[tStart:tEnd])
        self.axes ["S"].relim()
        self.axes ["S"].autoscale_view()
        self.axes ["S"].set_xlim((xStart,xEnd)) 
        
        self.lines["L2"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["L2"].set_ydata(self.L2[tStart:tEnd])
        self.axes ["L2"].relim()
        self.axes ["L2"].autoscale_view()
        self.axes ["L2"].set_xlim((xStart,xEnd)) 
        
        self.lines["L3"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["L3"].set_ydata(self.L3[tStart:tEnd])
        self.axes ["L3"].relim()
        self.axes ["L3"].autoscale_view()
        self.axes ["L3"].set_xlim((xStart,xEnd)) 
        
        self.lines["L4"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["L4"].set_ydata(self.L4[tStart:tEnd])
        self.axes ["L4"].relim()
        self.axes ["L4"].autoscale_view()
        self.axes ["L4"].set_xlim((xStart,xEnd)) 
        
        self.lines["L5"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["L5"].set_ydata(self.L5[tStart:tEnd])
        self.axes ["L5"].relim()
        self.axes ["L5"].autoscale_view()
        self.axes ["L5"].set_xlim((xStart,xEnd)) 
        
        self.lines["L6"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["L6"].set_ydata(self.L6[tStart:tEnd])
        self.axes ["L6"].relim()
        self.axes ["L6"].autoscale_view()
        self.axes ["L6"].set_xlim((xStart,xEnd)) 
        
        self.lines["L8"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["L8"].set_ydata(self.L8[tStart:tEnd])
        self.axes ["L8"].relim()
        self.axes ["L8"].autoscale_view()
        self.axes ["L8"].set_xlim((xStart,xEnd)) 
        
        self.lines["fmin"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["fmin"].set_ydata(self.fmin[tStart:tEnd])
        self.axes ["fmin"].relim()
        self.axes ["fmin"].autoscale_view()
        self.axes ["fmin"].set_xlim((xStart,xEnd)) 
        
        self.lines["fmax"].set_xdata(self.grid.tGrid[tStart:tEnd])
        self.lines["fmax"].set_ydata(self.fmax[tStart:tEnd])
        self.axes ["fmax"].relim()
        self.axes ["fmax"].autoscale_view()
        self.axes ["fmax"].set_xlim((xStart,xEnd)) 
        
        
        plt.draw()
        plt.show(block=final)
        
        return self.figure
    
    
    def add_timepoint(self):
#         E  = self.hamiltonian.E_kin  + self.hamiltonian.E_pot  + self.potential.E
#         E0 = self.hamiltonian.E_kin0 + self.hamiltonian.E_pot0 + self.potential.E0
        
#         E  = self.hamiltonian.E_kin  - self.potential.E
#         E0 = self.hamiltonian.E_kin0 - self.potential.E0
        
        E0 = self.hamiltonian.E0
        E  = self.hamiltonian.E
        
        self.energy  [self.iTime] = (E - E0) / E0
        self.partnum [self.iTime] = self.distribution.N_error
        
        if np.abs(self.hamiltonian.P0) > self.eps:
            self.momentum[self.iTime] = self.hamiltonian.P_error
        else:
            self.momentum[self.iTime] = self.hamiltonian.P
        
        self.entropy [self.iTime] = self.distribution.S_error
        
        self.L2[self.iTime] = self.distribution.L2_error
        self.L3[self.iTime] = self.distribution.L3_error
        self.L4[self.iTime] = self.distribution.L4_error
        self.L5[self.iTime] = self.distribution.L5_error
        self.L6[self.iTime] = self.distribution.L6_error
        self.L8[self.iTime] = self.distribution.L8_error
        
        self.fmin[self.iTime] = self.distribution.fmin
        self.fmax[self.iTime] = self.distribution.fmax_error
        
        self.title.set_text('t = %1.2f' % (self.grid.tGrid[self.iTime]))
        
        self.iTime += 1
        
    
    def get_timerange(self):
        tStart = self.iTime - (self.nTime+1)
        tEnd   = self.iTime
        
        if tStart < self.iStart:
            tStart = self.iStart
        
        xStart = self.grid.tGrid[tStart]
        xEnd   = self.grid.tGrid[tStart+self.nTime]
        
        return tStart, tEnd, xStart, xEnd
    
