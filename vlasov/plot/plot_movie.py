'''
Created on Mar 21, 2012

@author: mkraus
'''

import numpy as np

from scipy.ndimage     import zoom, gaussian_filter
from scipy.interpolate import interp1d, interp2d

import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.ticker import ScalarFormatter, MaxNLocator


class PlotMovie(object):
    '''
    classdocs
    '''

    def __init__(self, grid, distribution, hamiltonian, potential, nTime=0, nPlot=1, ntMax=0, vMin=None, vMax=None, cMax=False, cFac=1.5, write=False, interpolate=False, deltaf=False):
        '''
        Constructor
        '''
        
        self.prefix = '_pyVlasov1D_'
        
        self.grid         = grid
        self.distribution = distribution
        self.hamiltonian  = hamiltonian
        self.potential    = potential
        self.interpolate  = interpolate
        self.deltaf       = deltaf
        
        self.dpi = 100
        
        
        if ntMax == 0:
            ntMax = self.grid.nt
        
        if nTime > 0 and nTime <= ntMax:
            self.nTime = nTime
        else:
            self.nTime = ntMax
        
        if self.nTime > 10000:
            self.nTime = 10000
        
        
        self.iTime = 0
        self.nPlot = nPlot
        self.cMax  = cMax
        self.cFac  = cFac
        self.write = write
        
        self.partnum   = np.zeros(grid.nt+1)
        self.enstrophy = np.zeros(grid.nt+1)
        self.entropy   = np.zeros(grid.nt+1)
        self.energy    = np.zeros(grid.nt+1)
        self.momentum  = np.zeros(grid.nt+1)
        
        self.f       = self.distribution.f_ext
        
        self.x       = np.zeros(grid.nx+1)
        self.x[0:-1] = self.grid.x
        self.x[  -1] = self.grid.xMin() + self.grid.xLength()
        self.v       = self.grid.v
        
        if self.deltaf:
            self.fmax = np.zeros_like(self.f)
            for j in range(len(self.v)):
                self.fmax[:,j] = np.exp( - 0.5 * self.v[j]**2 )
            self.fmax /= self.fmax[0,:].sum() * self.grid.hv
        
        
        self.xMin = self.x[0]
        self.xMax = self.x[-1]
        
        if vMax is not None:
            self.vMax = vMax
        else:
            self.vMax = self.v[-1]

        if vMin is not None:
            self.vMin = vMin
        else:
            self.vMin = self.v[0]
        

        # set up figure/window size
        self.figure = plt.figure(num=None, figsize=(14,9), dpi=self.dpi)
        
        # set up plot margins
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        plt.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.05)
        
        # set up plot title
        self.title = self.figure.text(0.5, 0.97, 't = 0.0' % (grid.t[self.iTime]), horizontalalignment='center') 
        
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
        gs = gridspec.GridSpec(3, 2,
                       height_ratios=[4,1,1]
                       )
        
        self.axes["f"] = plt.subplot(gs[0,0:2])
        self.axes["N"] = plt.subplot(gs[1,0])
        self.axes["E"] = plt.subplot(gs[2,0])
        self.axes["P"] = plt.subplot(gs[1,1])
#        self.axes["S"] = plt.subplot(gs[1,1])
        self.axes["L"] = plt.subplot(gs[2,1])
        
#        self.axes["f"] = plt.subplot2grid((4,4), (0, 0), colspan=2, rowspan=2)
#        self.axes["N"] = plt.subplot2grid((4,4), (2, 0), colspan=2)
#        self.axes["E"] = plt.subplot2grid((4,4), (3, 0), colspan=2)
#        self.axes["n"] = plt.subplot2grid((4,4), (0, 2), rowspan=2)
#        self.axes["p"] = plt.subplot2grid((4,4), (2, 2), rowspan=2)
        
        
        # get f plot size in pixels
        fbox = self.axes["f"].get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        self.nx  = fbox.width  * self.dpi
        self.nv  = fbox.height * self.dpi
        
        # interpolated grid
        self.xint = np.linspace(self.xMin, self.xMax, self.nx)
        self.vint = np.linspace(self.vMin, self.vMax, self.nv)

        
        # distribution function (filled contour)
        self.axes ["f"].set_title('$f (x,v)$')
#        self.cbars["f"] = plt.colorbar(self.conts["f1"], orientation='horizontal')
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()

        # error in total particle number (time trace)
        self.lines["N"], = self.axes["N"].plot(self.grid.t[tStart:tEnd], self.partnum[tStart:tEnd])
        self.axes ["N"].set_title('$\Delta N (t)$')
        self.axes ["N"].set_xlim((xStart,xEnd)) 
        self.axes ["N"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["N"].yaxis.set_major_locator(MaxNLocator(4))
        
        
        # error in total enstrophy (time trace)
        self.lines["L"], = self.axes["L"].plot(self.grid.t[tStart:tEnd], self.enstrophy[tStart:tEnd])
        self.axes ["L"].set_title('$\Delta L_{2} (t)$')
        self.axes ["L"].set_xlim((xStart,xEnd)) 
        self.axes ["L"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["L"].yaxis.set_major_locator(MaxNLocator(4))
        
        # error in total energy (time trace)
        self.lines["E"], = self.axes["E"].plot(self.grid.t[tStart:tEnd], self.energy[tStart:tEnd])
        self.axes ["E"].set_title('$\Delta E (t)$')
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        self.axes ["E"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["E"].yaxis.set_major_locator(MaxNLocator(4))
        
        # error in total momentum (time trace)
        self.lines["P"], = self.axes["P"].plot(self.grid.t[tStart:tEnd], self.momentum[tStart:tEnd])
        if self.hamiltonian.P0 < 1E-8:
            self.axes ["P"].set_title('$P (t)$')
        else:
            self.axes ["P"].set_title('$\Delta P (t)$')
        self.axes ["P"].set_xlim((xStart,xEnd)) 
        self.axes ["P"].yaxis.set_major_formatter(majorFormatter)
        self.axes ["P"].yaxis.set_major_locator(MaxNLocator(4))
        
        # error in total entropy (time trace)
#        self.lines["S"], = self.axes["S"].plot(self.grid.t[tStart:tEnd], self.entropy[tStart:tEnd])
#        self.axes ["S"].set_title('$\Delta S (t)$')
#        self.axes ["S"].set_xlim((xStart,xEnd)) 
#        self.axes ["S"].yaxis.set_major_formatter(majorFormatter)
#        self.axes ["S"].yaxis.set_major_locator(MaxNLocator(4))
        
        
        self.update()
        
    
    def update_boundaries(self):
        fmin = +1e40
        fmax = -1e40
        
        
        if self.cMax and self.distribution.fMin != 0. and self.distribution.fMax != 0.:
            fmin = self.distribution.fMin
            fmax = self.distribution.fMax
        else:
            if self.deltaf:
                f  = self.distribution.f  - self.fmax[0:-1, :]
                f0 = self.distribution.f0 - self.fmax[0:-1, :]
#                 fl = self.distribution.fl - self.fmax[0:-1, :]
            else:
                f  = self.distribution.f
                f0 = self.distribution.f0
#                 fl = self.distribution.fl
            
            fmin = min(fmin, f.min(), f0.min() )
            fmax = max(fmax, f.max(), f0.max() )
            fmax *= self.cFac

        deltaf = (fmax-fmin)
        
        self.fnorm  = colors.Normalize(vmin=fmin-0.05*deltaf, vmax=fmax+0.01*deltaf)
        self.crange = np.linspace(-0.05*deltaf, fmax+0.01*deltaf, 100)
        
    
    def update(self, final=False):
        
        if not (self.iTime == 1 or (self.iTime-1) % self.nPlot == 0):
            return

#        self.update_boundaries()
        
#         for ckey, cont in self.conts.items():
#             for coll in cont.collections:
#                 self.axes[ckey].collections.remove(coll)
        
#         # contour plot
#         self.conts["f"] = self.axes["f"].contourf(self.x, self.v, self.f.T, 100, norm=self.fnorm, extend='neither')

#         # contour plot with Gaussian filter
#         fint = gaussian_filter(self.f.T, sigma=1.0, order=0)
#         self.conts["f"] = self.axes["f"].contourf(self.x, self.v, fint, 100, norm=self.fnorm, extend='neither')

#         # colormesh
#         self.axes["f"].cla()
#         self.axes["f"].pcolormesh(self.x, self.v, self.f.T, norm=self.fnorm)
        
        # colormesh with spline interpolation
        
        self.axes["f"].cla()
        
        if self.deltaf:
            f = (self.f - self.fmax).T
        else:
            f = self.f.T
        
        if self.interpolate:
            x = self.xint
            v = self.vint
        
            fspl = interp2d(self.x, self.v, f, kind='linear')
            f    = fspl(self.xint, self.vint)
            
        else:
            x = self.x
            v = self.v
        
        self.axes["f"].pcolormesh(x, v, f, norm=self.fnorm) # , shading='gouraud')
        
        self.axes["f"].set_xlim((self.xMin, self.xMax))
        self.axes["f"].set_ylim((self.vMin, self.vMax)) 
        
        
        tStart, tEnd, xStart, xEnd = self.get_timerange()
        
        self.lines["N"].set_xdata(self.grid.t[tStart:tEnd])
        self.lines["N"].set_ydata(self.partnum[tStart:tEnd])
        self.axes ["N"].relim()
        self.axes ["N"].autoscale_view()
        self.axes ["N"].set_xlim((xStart,xEnd)) 
        
        self.lines["L"].set_xdata(self.grid.t[tStart:tEnd])
        self.lines["L"].set_ydata(self.enstrophy[tStart:tEnd])
        self.axes ["L"].relim()
        self.axes ["L"].autoscale_view()
        self.axes ["L"].set_xlim((xStart,xEnd)) 
        
        self.lines["E"].set_xdata(self.grid.t[tStart:tEnd])
        self.lines["E"].set_ydata(self.energy[tStart:tEnd])
        self.axes ["E"].relim()
        self.axes ["E"].autoscale_view()
        self.axes ["E"].set_xlim((xStart,xEnd)) 
        
        self.lines["P"].set_xdata(self.grid.t[tStart:tEnd])
        self.lines["P"].set_ydata(self.momentum[tStart:tEnd])
        self.axes ["P"].relim()
        self.axes ["P"].autoscale_view()
        self.axes ["P"].set_xlim((xStart,xEnd)) 
        
#        self.lines["S"].set_xdata(self.grid.t[tStart:tEnd])
#        self.lines["S"].set_ydata(self.entropy[tStart:tEnd])
#        self.axes ["S"].relim()
#        self.axes ["S"].autoscale_view()
#        self.axes ["S"].set_xlim((xStart,xEnd))
        
        
        if self.write:
            filename = self.prefix + str('%06d' % (self.iTime-1)) + '.png'
            plt.savefig(filename, dpi=self.dpi)
        else:
            plt.draw()
            plt.show(block=final)
    
    
    def add_timepoint(self):
        self.partnum  [self.iTime] = self.distribution.N_error
        self.enstrophy[self.iTime] = self.distribution.L2_error
        self.entropy  [self.iTime] = self.distribution.S_error
        self.energy   [self.iTime] = self.hamiltonian.E_error
        
        if self.hamiltonian.P0 < 1E-8:
            self.momentum [self.iTime] = self.hamiltonian.P
        else:
            self.momentum [self.iTime] = self.hamiltonian.P_error
        
        
        self.title.set_text('t = %1.2f' % (self.grid.t[self.iTime]))
        
        self.iTime += 1
        
    
    def get_timerange(self):
        tStart = self.iTime - (self.nTime+1)
        if tStart < 0:
            tStart = 0
            
        tEnd = self.iTime
        
        xStart = self.grid.t[tStart]
        xEnd   = self.grid.t[tStart+self.nTime]
        
        return tStart, tEnd, xStart, xEnd
    
