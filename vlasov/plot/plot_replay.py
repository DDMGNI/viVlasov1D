'''
Created on Mar 21, 2012

@author: mkraus
'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter


class PlotReplay(object):
    '''
    classdocs
    '''

    def __init__(self, grid, distribution, hamiltonian, potential,nTime=0,  iStart=0, nPlot=1, vMax=0.0):
        '''
        Constructor
        '''
        
        # number of contour levels
        self.nconts = 20
        
        # colour map
        self.cmap = cm.jet
#         self.cmap = cm.hsv
#         self.cmap = cm.brg
#         self.cmap = cm.RdYlGn
        
        if nTime > 0 and nTime < grid.nt:
            self.nTime = nTime
        else:
            self.nTime = grid.nt
        
        self.nTime -= iStart
        self.iTime  = iStart
        self.iStart = iStart
        self.nPlot  = nPlot
        self.vMax   = vMax
        
        self.grid         = grid
        self.distribution = distribution
        self.hamiltonian  = hamiltonian
        self.potential    = potential
        
        self.partnum   = np.zeros(grid.nt+1)
        self.momentum  = np.zeros(grid.nt+1)
        self.energy    = np.zeros(grid.nt+1)
        self.enstrophy = np.zeros(grid.nt+1)
        self.entropy   = np.zeros(grid.nt+1)
        self.energy_f  = np.zeros(grid.nt+1)
        self.energy_p  = np.zeros(grid.nt+1)
        self.ekin      = np.zeros(grid.nt+1)
        self.epot      = np.zeros(grid.nt+1)
        
        self.x       = np.zeros(grid.nx+1)
        self.n       = np.zeros(grid.nx+1)
        self.phi     = np.zeros(grid.nx+1)
        
        self.x[0:-1] = self.grid.x
        self.x[  -1] = self.grid.xLength()
        
        
        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(16,9))
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.2, wspace=0.25)
        plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.05)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.97, 't = 0.0' % (self.grid.t[self.iTime]), horizontalalignment='center') 
        
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
        gs = gridspec.GridSpec(4, 4)
        
        self.axes["f"] = plt.subplot(gs[0:2,0:2])
        self.axes["n"] = plt.subplot(gs[0:2,2])
        self.axes["h"] = plt.subplot(gs[2:4,0:2])
        self.axes["p"] = plt.subplot(gs[2:4,2])
#         self.axes["T"] = plt.subplot(gs[0:2,3])
#         self.axes["V"] = plt.subplot(gs[2:4,3])
        self.axes["N"] = plt.subplot(gs[0,3])
        self.axes["P"] = plt.subplot(gs[1,3])
        self.axes["E"] = plt.subplot(gs[2,3])
        self.axes["L"] = plt.subplot(gs[3,3])
        
        
        # distribution function (filled contour)
        self.axes["f"] = plt.subplot(gs[0:2,0:2])
        self.axes ["f"].set_title('$f (x,v)$')
        self.conts["f"] = self.axes["f"].contourf(self.grid.x, self.grid.v, self.distribution.f.T, self.nconts, cmap=self.cmap, norm=self.fnorm)
        self.cbars["f"] = plt.colorbar(self.conts["f"], orientation='vertical')
        
        # Hamilton function (filled contour)
        self.axes["h"] = plt.subplot(gs[2:4,0:2])
        self.axes["h"].set_title('$H (x,v)$')
        self.conts["h"] = self.axes["h"].contourf(self.grid.x, self.grid.v, self.hamiltonian.h.T,  self.nconts, norm=self.hnorm)
        self.cbars["h"] = plt.colorbar(self.conts["h"], orientation='vertical')

        # density profile
        self.lines["n"], = self.axes["n"].plot(self.x, self.n)
#        self.axes ["n"].axis([self.grid.xMin, self.grid.xMax, self.density_min, self.density_max])
        self.axes ["n"].set_title('$n (x)$')
#        self.axes ["n"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["n"].set_xlim((0.0, self.grid.xLength())) 

        # potential profile
        self.lines["p"], = self.axes["p"].plot(self.x, self.phi)
#        self.axes ["p"].axis([self.grid.xMin, self.grid.xMax, self.potential_min, self.potential_max])
        self.axes ["p"].set_title('$\phi (x)$')
        self.axes ["p"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["p"].set_xlim((0.0, self.grid.xLength())) 
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()

#         # kinetic energy (time trace)
#         self.lines["T"], = self.axes["T"].plot(self.grid.t[tStart:tEnd], self.ekin[tStart:tEnd])
#         self.axes ["T"].set_title('$E_{kin} (t)$')
#         self.axes ["T"].set_xlim((xStart,xEnd)) 
# #        self.axes ["T"].yaxis.set_major_formatter(majorFormatter)
#         
#         # potential energy (time trace)
#         self.lines["V"], = self.axes["V"].plot(self.grid.t[tStart:tEnd], self.epot[tStart:tEnd])
#         self.axes ["V"].set_title('$E_{pot} (t)$')
#         self.axes ["V"].set_xlim((xStart,xEnd)) 
# #        self.axes ["V"].yaxis.set_major_formatter(majorFormatter)
        
        
        self.lines["N"], = self.axes["N"].plot(self.grid.t[tStart:tEnd], self.partnum  [tStart:tEnd])
        self.lines["P"], = self.axes["P"].plot(self.grid.t[tStart:tEnd], self.momentum [tStart:tEnd])
        self.lines["E"], = self.axes["E"].plot(self.grid.t[tStart:tEnd], self.energy   [tStart:tEnd])
        self.lines["L"], = self.axes["L"].plot(self.grid.t[tStart:tEnd], self.enstrophy[tStart:tEnd])
#        self.lines["E_f"], = self.axes["E"].plot(self.grid.t[tStart:tEnd], self.energy_f [tStart:tEnd])
#        self.lines["E_p"], = self.axes["E"].plot(self.grid.t[tStart:tEnd], self.energy_p [tStart:tEnd])
        
        self.axes ["N"].set_title('$\Delta N (t)$')
        if np.abs(self.hamiltonian.P0) < 1E-3: 
            self.axes ["P"].set_title('$P (t)$')
        else:
            self.axes ["P"].set_title('$\Delta P (t)$')
        self.axes ["E"].set_title('$\Delta E (t)$')
        self.axes ["L"].set_title('$\Delta L_{2} (t)$')

        self.axes ["N"].set_xlim((xStart,xEnd)) 
        self.axes ["P"].set_xlim((xStart,xEnd)) 
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        self.axes ["L"].set_xlim((xStart,xEnd)) 
        
        self.axes ["N"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["P"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["E"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["L"].yaxis.set_major_formatter(majorFormatter)
        
        
        # switch off some ticks
        plt.setp(self.axes["f"].get_xticklabels(), visible=False)
        plt.setp(self.axes["n"].get_xticklabels(), visible=False)
#         plt.setp(self.axes["T"].get_xticklabels(), visible=False)
        plt.setp(self.axes["N"].get_xticklabels(), visible=False)
        plt.setp(self.axes["P"].get_xticklabels(), visible=False)
        plt.setp(self.axes["E"].get_xticklabels(), visible=False)
        
        
        self.update()
        
        
        
    def save_plots(self):
        filename = str('F_%06d' % self.iTime) + '.png'
        extent = self.axes["f"].get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        self.figure.savefig(filename, dpi=70, bbox_inches=extent)

        filename = str('N_%06d' % self.iTime) + '.png'
        extent = self.axes["N"].get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        self.figure.savefig(filename, dpi=70, bbox_inches=extent)

        filename = str('P_%06d' % self.iTime) + '.png'
        extent = self.axes["P"].get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        self.figure.savefig(filename, dpi=70, bbox_inches=extent)

        filename = str('E_%06d' % self.iTime) + '.png'
        extent = self.axes["E"].get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        self.figure.savefig(filename, dpi=70, bbox_inches=extent)

        filename = str('L2_%06d' % self.iTime) + '.png'
        extent = self.axes["L"].get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        self.figure.savefig(filename, dpi=70, bbox_inches=extent)


    
    def update_boundaries(self):
        self.fmin = +1e40
        self.fmax = -1e40
        
        self.fmin = min(self.fmin, self.distribution.f.min() )
        self.fmax = max(self.fmax, self.distribution.f.max() )


        self.hmin = +1e40
        self.hmax = -1e40
        
        self.hmin = min(self.hmin, self.hamiltonian.h.min() )
        self.hmax = max(self.hmax, self.hamiltonian.h.max() )

        
        df = self.fmax - self.fmin
        
        self.fnorm = colors.Normalize(vmin=self.fmin + 0.05 * df, vmax=self.fmax + 0.05*df)
        self.hnorm = colors.Normalize(vmin=self.hmin, vmax=self.hmax)
        
        self.density_min   =-1.0
        self.density_max   = 1.5 * self.distribution.density.max()
        
#        self.potential_min = 1.5 * self.potential.phi.min()
#        self.potential_max = 1.5 * self.potential.phi.max()
#        
#        if self.potential_min == self.potential_max:
#            self.potential_min -= 1.0
#            self.potential_max += 1.0
        
    
    def update(self, final=False):
        
        if not (self.iTime == 1 or (self.iTime-1) % self.nPlot == 0 or self.iTime-1 == self.nTime):
            return
        
#        self.update_boundaries()

        for ckey, cont in self.conts.items():
            for coll in cont.collections:
                self.axes[ckey].collections.remove(coll)
        
#        self.fnorm = colors.Normalize(vmin=self.fmin, vmax=self.fmax)
        
        self.conts["f"] = self.axes["f"].contourf(self.grid.x, self.grid.v, self.distribution.f.T, self.nconts, cmap=self.cmap, norm=self.fnorm)
        self.conts["h"] = self.axes["h"].contourf(self.grid.x, self.grid.v, self.hamiltonian.h.T,  self.nconts, norm=self.hnorm)
        
        
        self.n  [0:-1] = self.distribution.density
        self.n  [  -1] = self.distribution.density[0]
        self.phi[0:-1] = self.potential.phi
        self.phi[  -1] = self.potential.phi[0]
        
        self.lines["n"].set_ydata(self.n)
        self.axes ["n"].relim()
        self.axes ["n"].autoscale_view()
        
        self.lines["p"].set_ydata(self.phi)
        self.axes ["p"].relim()
        self.axes ["p"].autoscale_view()
        
        
        if self.vMax > 0.0:
            self.axes["f"].set_ylim((-self.vMax, +self.vMax)) 
            
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()
        
#         self.lines["T"].set_xdata(self.grid.t[tStart:tEnd])
#         self.lines["T"].set_ydata(self.ekin[tStart:tEnd])
#         self.axes ["T"].relim()
#         self.axes ["T"].autoscale_view()
#         self.axes ["T"].set_xlim((xStart,xEnd)) 
#         
#         self.lines["V"].set_xdata(self.grid.t[tStart:tEnd])
#         self.lines["V"].set_ydata(self.epot[tStart:tEnd])
#         self.axes ["V"].relim()
#         self.axes ["V"].autoscale_view()
#         self.axes ["V"].set_xlim((xStart,xEnd)) 
        
        self.lines["N"].set_xdata(self.grid.t[tStart:tEnd])
        self.lines["N"].set_ydata(self.partnum[tStart:tEnd])
        self.axes ["N"].relim()
        self.axes ["N"].autoscale_view()
        self.axes ["N"].set_xlim((xStart,xEnd)) 
        
        self.lines["P"].set_xdata(self.grid.t[tStart:tEnd])
        self.lines["P"].set_ydata(self.momentum[tStart:tEnd])
        self.axes ["P"].relim()
        self.axes ["P"].autoscale_view()
        self.axes ["P"].set_xlim((xStart,xEnd)) 
        
        self.lines["E"].set_xdata(self.grid.t[tStart:tEnd])
        self.lines["E"].set_ydata(self.energy[tStart:tEnd])
        self.axes ["E"].relim()
        self.axes ["E"].autoscale_view()
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        
        self.lines["L"].set_xdata(self.grid.t[tStart:tEnd])
        self.lines["L"].set_ydata(self.enstrophy[tStart:tEnd])
        self.axes ["L"].relim()
        self.axes ["L"].autoscale_view()
        self.axes ["L"].set_xlim((xStart,xEnd)) 
        
        
        plt.draw()
        plt.show(block=final)
        
        return self.figure
    
    
    def add_timepoint(self):
#         E  = self.hamiltonian.E_kin  + self.hamiltonian.E_pot  + self.potential.E
#         E0 = self.hamiltonian.E_kin0 + self.hamiltonian.E_pot0 + self.potential.E0
        
        E0 = self.hamiltonian.E0
        E  = self.hamiltonian.E
        
        if np.abs(self.hamiltonian.P0) < 1E-3: 
            self.momentum[self.iTime] = self.hamiltonian.P
        else:
            self.momentum[self.iTime] = self.hamiltonian.P_error
        
        self.energy [self.iTime] = (E - E0) / E0
        self.partnum  [self.iTime] = self.distribution.N_error
        self.enstrophy[self.iTime] = self.distribution.L2_error
        self.entropy  [self.iTime] = self.distribution.S_error
        
        self.title.set_text('t = %1.2f' % (self.grid.t[self.iTime]))
        
        self.iTime += 1
        
    
    def get_timerange(self):
        tStart = self.iTime - (self.nTime+1)
        tEnd   = self.iTime
        
        if tStart < self.iStart:
            tStart = self.iStart
        
        xStart = self.grid.t[tStart]
        xEnd   = self.grid.t[tStart+self.nTime]
        
        return tStart, tEnd, xStart, xEnd
    
