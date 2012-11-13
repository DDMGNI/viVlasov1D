'''
Created on Nov 09, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

from core import Config
from data import maxwellian

from vlasov.predictor.PETScArakawaRK4   import PETScArakawaRK4
from vlasov.predictor.PETScPoisson      import PETScPoissonSolver
from vlasov.predictor.PETScVlasovSolver import PETScVlasovSolver


class petscVP1Dbase(object):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        # load run config file
        cfg = Config(cfgfile)
        
        # timestep setup
        self.ht    = cfg['grid']['ht']              # timestep size
        self.nt    = cfg['grid']['nt']              # number of timesteps
        self.nsave = cfg['io']['nsave']             # save only every nsave'th timestep
        
        # grid setup
        self.nx = cfg['grid']['nx']                 # number of points in x
        self.nv = cfg['grid']['nv']                 # number of points in v
        L       = cfg['grid']['L']
        vMin    = cfg['grid']['vmin']
        vMax    = cfg['grid']['vmax']
        
        self.hx = L / self.nx                       # gridstep size in x
        self.hv = (vMax - vMin) / (self.nv-1)       # gridstep size in v
        
        self.time = PETSc.Vec().createMPI(1, 1, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        self.poisson = cfg['solver']['poisson_const']     # Poisson constant
        self.alpha   = cfg['solver']['alpha']             # collision constant
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_rtol', cfg['solver']['petsc_residual'])
        OptDB.setValue('ksp_max_it', cfg['solver']['petsc_max_iter'])

#        OptDB.setValue('ksp_monitor', '')
#        OptDB.setValue('log_info', '')
#        OptDB.setValue('log_summary', '')
        
        
        # create DA with single dof
        self.da1 = PETSc.DA().create(dim=2, dof=1,
                                    sizes=[self.nx, self.nv],
                                    proc_sizes=[PETSc.COMM_WORLD.getSize(), 1],
                                    boundary_type=('periodic', 'none'),
                                    stencil_width=1,
                                    stencil_type='box')
        
        # create DA (dof = number of species + 1 for the potential)
        self.da2 = PETSc.DA().create(dim=2, dof=2,
                                     sizes=[self.nx, self.nv],
                                     proc_sizes=[PETSc.COMM_WORLD.getSize(), 1],
                                     boundary_type=('periodic', 'none'),
                                     stencil_width=1,
                                     stencil_type='box')
        
        
        # create DA for Poisson guess
        self.dax = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[self.nx],
                                    proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                    boundary_type=('periodic'),
                                    stencil_width=1,
                                    stencil_type='box')
        
        # create DA for y grid
        self.day = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[self.nv],
                                    proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                    boundary_type=('none'))
        
        
        # initialise grid
        self.da1.setUniformCoordinates(xmin=0.0,  xmax=L,
                                       ymin=vMin, ymax=vMax)
        
        self.da2.setUniformCoordinates(xmin=0.0,  xmax=L,
                                       ymin=vMin, ymax=vMax)
        
        self.dax.setUniformCoordinates(xmin=0.0, xmax=L)
        
        self.day.setUniformCoordinates(xmin=vMin, xmax=vMax) 
        
        
        # get coordinate vectors
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
        
        # save x coordinate arrays
        scatter, xVec = PETSc.Scatter.toAll(coords_y)

        scatter.begin(coords_y, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_y, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                  
        self.xGrid = xVec.getValues(range(0, self.nx)).copy()
        
        scatter.destroy()
        xVec.destroy()
        
        # save v coordinate arrays
        scatter, vVec = PETSc.Scatter.toAll(coords_y)

        scatter.begin(coords_y, vVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_y, vVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                  
        self.vGrid = vVec.getValues(range(0, self.nv)).copy()
        
        scatter.destroy()
        vVec.destroy()
        
        
        # create solution and RHS vector
        self.x  = self.da2.createGlobalVec()
        self.b  = self.da2.createGlobalVec()
        
        # create solution and RHS vector for Vlasov and Poisson solver
        self.fb = self.da1.createGlobalVec()
        self.pb = self.dax.createGlobalVec()
        
        # create vectors for Hamiltonians, distribution functions,
        # density and the potential
        self.h0 = self.da1.createGlobalVec()
        self.h1 = self.da1.createGlobalVec()
        self.f  = self.da1.createGlobalVec()
        self.n  = self.dax.createGlobalVec()
        self.p  = self.dax.createGlobalVec()
        
        # set variable names
        self.x.setName('solver_x')
        self.b.setName('solver_b')
        
        self.h0.setName('h0')
        self.h1.setName('h1')
        self.f.setName('f')
        self.n.setName('n')
        self.p.setName('phi')
        
        
        # create Vlasov matrix and solver
        self.vlasov_mat = PETScVlasovSolver(self.da1, self.h0, 
                                            self.nx, self.nv, self.ht, self.hx, self.hv)
        
        self.vlasov_A = PETSc.Mat().createPython([self.f.getSizes(), self.fb.getSizes()], comm=PETSc.COMM_WORLD)
        self.vlasov_A.setPythonContext(self.vlasov_mat)
        self.vlasov_A.setUp()
        
        self.vlasov_ksp = PETSc.KSP().create()
        self.vlasov_ksp.setFromOptions()
        self.vlasov_ksp.setOperators(self.vlasov_A)
        self.vlasov_ksp.setType('gmres')
        self.vlasov_ksp.getPC().setType('none')
        self.vlasov_ksp.setInitialGuessNonzero(True)
        
        
        # create Poisson matrix and solver
        self.poisson_mat = PETScPoissonSolver(self.da1, self.dax, 
                                              self.nx, self.nv, self.hx, self.hv,
                                              self.poisson)
        
        self.poisson_A = PETSc.Mat().createPython([self.p.getSizes(), self.pb.getSizes()], comm=PETSc.COMM_WORLD)
        self.poisson_A.setPythonContext(self.poisson_mat)
        self.poisson_A.setUp()
        
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_A)
        self.poisson_ksp.setType('cg')
        self.poisson_ksp.getPC().setType('none')
#        self.poisson_ksp.setInitialGuessNonzero(True)
        
        
        # create Arakawa RK4 solver object
        self.arakawa_rk4 = PETScArakawaRK4(self.da1, self.h0, self.nx, self.nv, self.ht, self.hx, self.hv)
        
        
        # set initial data
        n0 = self.dax.createGlobalVec()
        T0 = self.dax.createGlobalVec()
        
        n0_arr = self.dax.getVecArray(n0)
        T0_arr = self.dax.getVecArray(T0)
        
        n0.setName('n0')
        T0.setName('T0')
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print
            print("ht = %e" % (self.ht))
            print("hx = %e" % (self.hx))
            print("hv = %e" % (self.hv))
            print("a  = %e" % (self.alpha))
            print
            print("CFL = %e" % (self.hx / vMax))
            print
        
        
        f_arr = self.da1.getVecArray(self.f)
        x_arr = self.da2.getVecArray(self.x)
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        coords  = self.da2.getCoordinateDA().getVecArray(self.da2.getCoordinates())
        
        if cfg['initial_data']['distribution_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['distribution_python'], globals(), locals(), ['distribution'], 0)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    if j == 0 or j == self.nv-1:
                        f_arr[i,j] = 0.0
                    else:
                        f_arr[i,j] = init_data.distribution(coords[i,j][0], coords[i,j][1]) 
            
            n0_arr[xs:xe] = 0.
            T0_arr[xs:xe] = 0.
        
        else:
            if cfg['initial_data']['density_python'] != None:
                init_data = __import__("runs." + cfg['initial_data']['density_python'], globals(), locals(), ['distribution'], 0)
                
                for i in range(xs, xe):
                    n0_arr[i] = init_data.density(coords[i,0][0], L) 
            
            else:
                n0_arr[xs:xe] = cfg['initial_data']['density']            
            
            
            if cfg['initial_data']['temperature_python'] != None:
                init_data = __import__("runs." + cfg['initial_data']['temperature_python'], globals(), locals(), ['distribution'], 0)
                
                for i in range(xs, xe):
                    T0_arr[i] = init_data.temperature(coords[i,0][0]) 
            
            else:
                T0_arr[xs:xe] = cfg['initial_data']['temperature']            
            
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    if j == 0 or j == self.nv-1:
                        f_arr[i,j] = 0.0
                    else:
                        f_arr[i,j] = n0_arr[i] * maxwellian(T0_arr[i], coords[i,j][1])
        
        # normalise f to fit density
        ### TODO (?) ###
        
        self.copy_f_to_x()                    # copy distribution function to solution vector
        self.calculate_density()              # calculate density
        self.calculate_potential()            # calculate initial potential
        
        
        # initialise kinetic hamiltonian
        h0_arr = self.da1.getVecArray(self.h0)
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                h0_arr[i, j] = 0.5 * coords[i,j][1]**2 # * self.mass
        
        
        # create HDF5 output file
        self.hdf5_viewer = PETSc.Viewer().createHDF5(cfg['io']['hdf5_output'],
                                          mode=PETSc.Viewer.Mode.WRITE,
                                          comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.HDF5PushGroup("/")
        
        
        # write grid data to hdf5 file
        coords_x.setName('x')
        coords_y.setName('v')

        self.hdf5_viewer(coords_x)
        self.hdf5_viewer(coords_y)
        
        # write initial data to hdf5 file
        self.hdf5_viewer(n0)
        self.hdf5_viewer(T0)
        
        self.hdf5_viewer.HDF5SetTimestep(0)
        self.save_hdf5_vectors()        
        
    
    def __del__(self):
        del self.hdf5_viewer
        
 

    def initial_guess(self):
        # calculate initial guess for distribution function
        self.arakawa_rk4.rk4(self.f, self.h1)
        self.copy_f_to_x()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     RK4")
        
        # calculate initial guess for potential
        self.calculate_potential()
        
        # correct initial guess for distribution function
        self.calculate_vlasov()
        
    
    def calculate_vlasov(self):
        self.vlasov_mat.update_potential(self.h1)
        self.vlasov_mat.formRHS(self.fb)
        self.vlasov_ksp.solve(self.fb, self.f)
        self.copy_f_to_x()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Vlasov:   %5i iterations,   residual = %24.16E " % (self.vlasov_ksp.getIterationNumber(), self.vlasov_ksp.getResidualNorm()) )
            print
        
        
    def calculate_potential(self):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.poisson_mat.formRHS(self.f, self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        p_arr  = self.dax.getVecArray(self.p)
        
        phisum = self.p.sum()
        phiave = phisum / self.nx

        p_arr[xs:xe] -= phiave
        
        self.copy_p_to_x()
        self.copy_p_to_h()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Poisson:  %5i iterations,   residual = %24.16E" % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
            print("                                   sum(phi) = %24.16E" % (phisum))
    
    
    def calculate_density(self):
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        # copy solution to f and p vectors
        f_arr  = self.da1.getVecArray(self.f)
        n_arr  = self.dax.getVecArray(self.n)
        
        n_arr[xs:xe] = f_arr[xs:xe, :].sum(axis=1) * self.hv
    
    
    def copy_x_to_f(self):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        x_arr  = self.da2.getVecArray(self.x)
        f_arr  = self.da1.getVecArray(self.f)
        
        f_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 0] 
        
    
    def copy_f_to_x(self):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        x_arr  = self.da2.getVecArray(self.x)
        f_arr  = self.da1.getVecArray(self.f)
        
        x_arr[xs:xe, ys:ye, 0] = f_arr[xs:xe, ys:ye] 
    
    
    def copy_x_to_p(self):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        x_arr  = self.da2.getVecArray(self.x)
        p_arr  = self.dax.getVecArray(self.p)
        
        p_arr[xs:xe] = x_arr[xs:xe, 0, 1]
        
    
    def copy_p_to_x(self):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        p_arr  = self.dax.getVecArray(self.p)
        x_arr  = self.da2.getVecArray(self.x)
        
        for j in range(ys, ye):
            x_arr [xs:xe, j, 1] = p_arr[xs:xe]
        
        
    def copy_p_to_h(self):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        p_arr  = self.dax.getVecArray(self.p)
        h1_arr = self.da1.getVecArray(self.h1)
    
        for j in range(ys, ye):
            h1_arr[xs:xe, j] = p_arr[xs:xe]
        

    def save_to_hdf5(self, itime):
        
        self.calculate_density()
        
        # save to hdf5 file
        if itime % self.nsave == 0 or itime == self.nt + 1:
            self.hdf5_viewer.HDF5SetTimestep(self.hdf5_viewer.HDF5GetTimestep() + 1)
            self.save_hdf5_vectors()


    def save_hdf5_vectors(self):
        self.hdf5_viewer(self.time)
        self.hdf5_viewer(self.x)
        self.hdf5_viewer(self.b)
        self.hdf5_viewer(self.f)
        self.hdf5_viewer(self.n)
        self.hdf5_viewer(self.p)
        self.hdf5_viewer(self.h0)
        self.hdf5_viewer(self.h1)
