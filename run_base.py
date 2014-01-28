'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, time, datetime

import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

from vlasov.core.config  import Config

from vlasov.toolbox.Grid    import Grid
from vlasov.toolbox.VIDA    import VIDA
from vlasov.toolbox.Toolbox import Toolbox

from vlasov.explicit.PETScArakawaRungeKutta import PETScArakawaRungeKutta
from vlasov.explicit.PETScArakawaGear       import PETScArakawaGear
from vlasov.explicit.PETScArakawaSymplectic import PETScArakawaSymplectic

# from vlasov.solvers.poisson.PETScPoissonSolver2  import PETScPoissonSolver
from vlasov.solvers.poisson.PETScPoissonSolver4  import PETScPoissonSolver


class petscVP1Dbase():
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile, runid):
        '''
        Constructor
        '''
        
        # if runid is empty use timestamp
        if runid == "":
            runid = datetime.datetime.fromtimestamp(time.time()).strftime("%y%m%d%H%M%S")
        
        # stencil width
        stencil = 2
        
        # load run config file
        self.cfg = Config(cfgfile)
        
        # timestep setup
        ht         = self.cfg['grid']['ht']              # timestep size
        nt         = self.cfg['grid']['nt']              # number of timesteps
        self.nsave = self.cfg['io']['nsave']             # save only every nsave'th timestep
        
        # grid setup
        nx   = self.cfg['grid']['nx']                 # number of points in x
        nv   = self.cfg['grid']['nv']                 # number of points in v
        L    = self.cfg['grid']['L']
        vMax = self.cfg['grid']['vmax']
        vMin = self.cfg['grid']['vmin']
        
        if vMin == 0.: 
            vMin = -vMax
        
        Lx = L
        Lv = vMax - vMin
        
        hx = Lx / nx                      # gridstep size in x
        hv = Lv / (nv-1)                  # gridstep size in v
        
        self.time = PETSc.Vec().createMPI(1, 1, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        
        self.solver_package = self.cfg['solver']['lu_package']
        
        self.nInitial  = self.cfg['solver']['initial_iter']          # number of iterations for initial guess
        self.coll_freq = self.cfg['solver']['coll_freq']             # collision frequency
        
        self.charge = self.cfg['initial_data']['charge']             # particle charge
        self.mass   = self.cfg['initial_data']['mass']               # particle mass
        
        output_directory = self.cfg['io']['output_dir']
        
        if output_directory == None or output_directory == "":
            output_directory = "."
        
        tindex = cfgfile.rfind('/')
        run_filename     = cfgfile[tindex:].replace('.cfg', '.') + runid
        hdf_out_filename = output_directory + '/' + run_filename + ".hdf5"
        cfg_out_filename = output_directory + '/' + run_filename + ".cfg" 

#         hdf_in_filename  = self.cfg['io']['hdf5_input']
#         hdf_out_filename = self.cfg['io']['hdf5_output']
        
        self.cfg.write_current_config(cfg_out_filename)
        
        
        # set initial guess method
        initial_guess_options = {
            None          : self.initial_guess_none,
            ""            : self.initial_guess_none,
            "rk4"         : self.initial_guess_rk4,
            "gear"        : self.initial_guess_gear,
            "symplectic2" : self.initial_guess_symplectic2,
            "symplectic4" : self.initial_guess_symplectic4,
        }
        
        self.initial_guess_method = initial_guess_options[self.cfg['solver']['initial_guess']] 
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        OptDB.setValue('ksp_atol',   self.cfg['solver']['petsc_ksp_atol'])
        OptDB.setValue('ksp_max_it', self.cfg['solver']['petsc_ksp_max_iter'])

        OptDB.setValue('snes_rtol',   self.cfg['solver']['petsc_snes_rtol'])
        OptDB.setValue('snes_atol',   self.cfg['solver']['petsc_snes_atol'])
        OptDB.setValue('snes_stol',   self.cfg['solver']['petsc_snes_stol'])
        OptDB.setValue('snes_max_it', self.cfg['solver']['petsc_snes_max_iter'])
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Initialising Distributed Arrays.")
            
        # create DA for 2d grid (f only)
#         self.da1 = VIDA().create(dim=2, dof=1,
#                                        sizes=[nx, nv],
#                                        proc_sizes=[PETSc.COMM_WORLD.getSize(), 1],
#                                        boundary_type=['periodic', 'ghosted'],
#                                        stencil_width=stencil,
#                                        stencil_type='box')
        self.da1 = VIDA().create(dim=2, dof=1,
                                       sizes=[nx, nv],
                                       proc_sizes=[1, PETSc.COMM_WORLD.getSize()],
                                       boundary_type=['periodic', 'ghosted'],
                                       stencil_width=stencil,
                                       stencil_type='box')
#         self.da1 = VIDA().create(dim=2, dof=1,
#                                        sizes=[nx, nv],
#                                        proc_sizes=[PETSc.DECIDE, 2],
#                                        boundary_type=['periodic', 'ghosted'],
#                                        stencil_width=stencil,
#                                        stencil_type='box')
#         self.da1 = VIDA().create(dim=2, dof=1,
#                                        sizes=[nx, nv],
#                                        proc_sizes=[PETSc.DECIDE, PETSc.DECIDE],
#                                        boundary_type=['periodic', 'ghosted'],
#                                        stencil_width=stencil,
#                                        stencil_type='box')
        
        # create VIDA for x grid
        self.dax = VIDA().create(dim=1, dof=1,
                                       sizes=[nx],
                                       proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                       boundary_type=('periodic'),
                                       stencil_width=stencil,
                                       stencil_type='box')
        
        # create VIDA for y grid
        self.day = VIDA().create(dim=1, dof=1,
                                       sizes=[nv],
                                       proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                       boundary_type=('ghosted'),
                                       stencil_width=stencil,
                                       stencil_type='box')
        
        
        # initialise grid
        self.da1.setUniformCoordinates(xmin=0.0,  xmax=Lx, ymin=vMin, ymax=vMax)
        self.dax.setUniformCoordinates(xmin=0.0,  xmax=Lx)
        self.day.setUniformCoordinates(xmin=vMin, xmax=vMax) 
        
        # get local index ranges
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # get coordinate vectors
        coords_x = self.dax.getCoordinates()
        coords_v = self.day.getCoordinates()
        
        # save x coordinate arrays
        scatter, xVec = PETSc.Scatter.toAll(coords_x)
 
        scatter.begin(coords_x, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_x, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
         
        xGrid = xVec.getValues(range(0, nx)).copy()
         
        scatter.destroy()
        xVec.destroy()
         
        # save v coordinate arrays
        scatter, vVec = PETSc.Scatter.toAll(coords_v)
 
        scatter.begin(coords_v, vVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_v, vVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                   
        vGrid = vVec.getValues(range(0, nv)).copy()
         
        scatter.destroy()
        vVec.destroy()
        
        
        # create grid object
        self.grid = Grid(xGrid, vGrid, nt, nx, nv, ht, hx, hv, stencil)
        
        
        # create vectors for Hamiltonians
        self.h0  = self.da1.createGlobalVec()        # kinetic Hamiltonian
        self.h1c = self.da1.createGlobalVec()        # current  potential Hamiltonian
        self.h2c = self.da1.createGlobalVec()        # current  external  Hamiltonian
        self.h1h = self.da1.createGlobalVec()        # previous potential Hamiltonian
        self.h2h = self.da1.createGlobalVec()        # previous external  Hamiltonian
        
        # distribution functions
        self.fl     = self.da1.createGlobalVec()     # last    (k+1, n  )
        self.fc     = self.da1.createGlobalVec()     # current (k+1, n+1)
        self.fh     = self.da1.createGlobalVec()     # history (k)
        
        # distribution function solver vectors
        self.fb     = self.da1.createGlobalVec()     # right hand side
        
        # moments
        self.N      = self.dax.createGlobalVec()     # density
        self.U      = self.dax.createGlobalVec()     # velocity density
        self.E      = self.dax.createGlobalVec()     # energy density
        self.A      = self.dax.createGlobalVec()     # collision factor
        
        # local moments
        self.nc     = PETSc.Vec().createSeq(nx)        # current density
        self.uc     = PETSc.Vec().createSeq(nx)        # current velocity density
        self.ec     = PETSc.Vec().createSeq(nx)        # current energy density
        self.ac     = PETSc.Vec().createSeq(nx)        # current collision factor
        self.nh     = PETSc.Vec().createSeq(nx)        # history density
        self.uh     = PETSc.Vec().createSeq(nx)        # history velocity density
        self.eh     = PETSc.Vec().createSeq(nx)        # history energy density
        self.ah     = PETSc.Vec().createSeq(nx)        # history collision factor
        
        # internal potential
        self.pc_int = self.dax.createGlobalVec()     # current
        self.ph_int = self.dax.createGlobalVec()     # history
        
        # external potential
        self.pc_ext = self.dax.createGlobalVec()    # current
        self.ph_ext = self.dax.createGlobalVec()    # history
        
        # potential solver vectors
        self.pb     = self.dax.createGlobalVec()     # right hand side
        self.pn     = self.dax.createGlobalVec()     # null vector
        
        # set variable names
        self.h0.setName('h0')
        self.h1c.setName('h1')
        self.h2c.setName('h2')
        self.fc.setName('f')
        self.pc_int.setName('phi_int')
        self.pc_ext.setName('phi_ext')
        
        self.N.setName('n')
        self.U.setName('u')
        self.E.setName('e')
        
        
        # initialise nullspace basis vectors
        # the Poisson equation has a null space of all constant vectors
        # that needs to be removed to avoid jumpy potentials
        self.pn.set(1.)
        self.pn.normalize()
        
        self.p_nullspace = PETSc.NullSpace().create(constant=False, vectors=(self.pn,))
        
        
        # create Toolbox
        self.toolbox = Toolbox(self.da1, self.dax, self.grid)
        
        
        # initialise kinetic hamiltonian
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Initialising kinetic Hamiltonian.")
        self.toolbox.initialise_kinetic_hamiltonian(self.h0, self.mass)
                
        
        # create Arakawa initial guess solver object
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Instantiating Initial Guess Objects.")
            
        self.arakawa_rk4        = PETScArakawaRungeKutta(self.da1, self.grid, self.h0, self.h1h, self.h2h, self.nInitial)
        self.arakawa_gear       = PETScArakawaGear      (self.da1, self.grid, self.h0, self.h1h, self.h2h, self.nInitial)
        self.arakawa_symplectic = PETScArakawaSymplectic(self.da1, self.grid, self.h0, self.h1h, self.h2h, self.nInitial)
        
        
        # create Vlasov solver dummy
        self.vlasov_solver = None        
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Instantiating Poisson Object.")
        
        # initialise Poisson matrix
        self.poisson_matrix = self.dax.createMat()
        self.poisson_matrix.setUp()
        self.poisson_matrix.setNullSpace(self.p_nullspace)
        
        # create Poisson object
        self.poisson_solver = PETScPoissonSolver(self.dax, nx, hx, self.charge)
        self.poisson_solver.formMat(self.poisson_matrix)
        
        
        # create linear Poisson solver
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Create Poisson Solver.")
            
        OptDB.setValue('ksp_rtol', 1E-13)

#         OptDB.setValue('pc_gamg_type', 'agg')
#         OptDB.setValue('pc_gamg_agg_nsmooths', '1')
        
#         OptDB.setValue('pc_hypre_type', 'boomeramg')


        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_matrix)
#         self.poisson_ksp.setType('gmres')
        self.poisson_ksp.setType('cg')
#         self.poisson_ksp.setType('bcgs')
#         self.poisson_ksp.getPC().setType('none')
#         self.poisson_ksp.getPC().setType('gamg')
        self.poisson_ksp.getPC().setType('hypre')
#         self.poisson_ksp.getPC().setType('ml')
        
#         self.poisson_nsp = PETSc.NullSpace().create(vectors=(self.p_nvec,))
#         self.poisson_ksp.setNullSpace(self.poisson_nsp)
        
        
        OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print()
            print("Run ID:      %s" % runid)
            print()
            print("Config File: %s" % cfgfile)
            print("Output File: %s" % hdf_out_filename)
            print()
            print("nt = %i" % (self.grid.nt))
            print("nx = %i" % (self.grid.nx))
            print("nv = %i" % (self.grid.nv))
            print()
            print("ht = %e" % (self.grid.ht))
            print("hx = %e" % (self.grid.hx))
            print("hv = %e" % (self.grid.hv))
            print()
            print("xMin = %+12.6e" % (self.grid.xMin()))
            print("xMax = %+12.6e" % (self.grid.vMax()))
            print("vMin = %+12.6e" % (self.grid.vMin()))
            print("vMax = %+12.6e" % (self.grid.vMax()))
            print()
            print("nu   = %7.1e" % (self.coll_freq))
            print()
            print("CFL  = %e" % (self.grid.hx / vMax))
            print()
            print()
        
        
        # set initial data
        n0 = self.dax.createGlobalVec()
        T0 = self.dax.createGlobalVec()
        
        n0.setName('n0')
        T0.setName('T0')
        
        # !!! TODO !!! #
#         c_arr = self.da1.getVecArray(coords)
#         x_arr = self.dax.getGlobalArray(coords_x)
#         v_arr = self.day.getGlobalArray(coords_v)
#         print(coords[:])
        
        if self.cfg['initial_data']['distribution_python'] != None:
            init_data = __import__("runs." + self.cfg['initial_data']['distribution_python'], globals(), locals(), ['distribution'], 0)
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("Initialising distribution function with Python function.")
            
            self.toolbox.initialise_distribution_function(self.fc, init_data.distribution)
            
            n0.set(0.)
            T0.set(0.)
        
        else:
            n0_arr = self.dax.getVecArray(n0)
            T0_arr = self.dax.getVecArray(T0)
            
            if self.cfg['initial_data']['density_python'] != None:
                init_data = __import__("runs." + self.cfg['initial_data']['density_python'], globals(), locals(), ['density'], 0)
                
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("Initialising density with Python function.")
            
                for i in range(xs, xe):
                    n0_arr[i] = init_data.density(self.grid.x[i], self.grid.xLength()) 
            
            else:
                n0_arr[xs:xe] = self.cfg['initial_data']['density']            
            
            
            if self.cfg['initial_data']['temperature_python'] != None:
                init_data = __import__("runs." + self.cfg['initial_data']['temperature_python'], globals(), locals(), ['temperature'], 0)
                
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("Initialising temperature with Python function.")
            
                for i in range(xs, xe):
                    T0_arr[i] = init_data.temperature(self.grid.x[i]) 
            
            else:
                T0_arr[xs:xe] = self.cfg['initial_data']['temperature']            
            
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("Initialising distribution function with Maxwellian.")
            
            self.toolbox.initialise_distribution_nT(self.fc, n0, T0)
            
        
        # normalise f
        self.normalise_distribution_function()
        
        
        # calculate potential and moments
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Calculate initial potential and moments.")
        self.calculate_moments(output=False)
        
        
        # initialise Gear History
        if self.cfg['solver']['initial_guess'] == "gear":
            self.arakawa_gear.initialise_history(self.fc)

        
        # check for external potential
        if self.cfg['initial_data']['external_python'] != None:
            if PETSc.COMM_WORLD.getRank() == 0:
                print("Calculate external potential.")
                
            external_data = __import__("runs." + self.cfg['initial_data']['external_python'], globals(), locals(), ['external'], 0)
            self.external = external_data.external
        else:
            self.external = None
        
        # calculate external potential
        self.calculate_external(0.)
        
        
        # create HDF5 output file
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Create HDF5 output file.")
        
        self.hdf5_viewer = PETSc.ViewerHDF5().create(hdf_out_filename,
                                              mode=PETSc.Viewer.Mode.WRITE,
                                              comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.pushGroup("/")
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Saving initial data to HDF5.")
        
        # write grid data to hdf5 file
        coords_x.setName('x')
        coords_v.setName('v')

        self.hdf5_viewer(coords_x)
        self.hdf5_viewer(coords_v)
        
        # write initial data to hdf5 file
        self.hdf5_viewer(n0)
        self.hdf5_viewer(T0)
        
        # save to hdf5
        self.hdf5_viewer.setTimestep(0)
        self.save_hdf5_vectors()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("run_base.py: initialisation done.")
            print("")
    
    
    def normalise_distribution_function(self):
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Normalise distribution function.")
        
        nave = self.fc.sum() * self.grid.hv / self.grid.nx
        self.fc.scale(1./nave)

    
    def calculate_residual(self):
        self.vlasov_solver.function_mult(self.fc, self.fb)
        fnorm = self.fb.norm()
        
        self.poisson_solver.function_mult(self.pc_int, self.N, self.pb)
        pnorm = self.pb.norm()
        
        return fnorm + pnorm
    
    
    def calculate_moments(self, potential=True, output=True, f=None):
        if f == None: f = self.fc
        
        self.toolbox.compute_density(f, self.N)
        self.toolbox.compute_velocity_density(f, self.U)
        self.toolbox.compute_energy_density(f, self.E)
        self.toolbox.compute_collision_factor(self.N, self.U, self.E, self.A)
 
        self.copy_xvec_to_seq(self.N, self.nc)
        self.copy_xvec_to_seq(self.U, self.uc)
        self.copy_xvec_to_seq(self.E, self.ec)
        self.copy_xvec_to_seq(self.A, self.ac)
        
        if potential:
            self.calculate_potential(output)         # calculate initial potential
            self.copy_pint_to_h()                    # copy potential to Hamiltonian
    
    
    def calculate_potential(self, output=True):
        
        self.poisson_solver.formRHS(self.N, self.pb)
        self.poisson_ksp.solve(self.pb, self.pc_int)
        
        if output:
            phisum = self.pc.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Poisson:                               sum(phi) = %24.16E" % (phisum))
    
    
    def calculate_external(self, t, p=None):
        (xs, xe), = self.dax.getRanges()
        
        if p == None:
            p = self.pc_ext
        
        if self.external != None:
            p_ext_arr = self.dax.getVecArray(p)
            
            for i in range(xs, xe):
                p_ext_arr[i] = self.external(self.grid.x[i], t) 
            
            # remove average
            phisum = p.sum()
            phiave = phisum / self.grid.nx
            p.shift(-phiave)
            
        if p == self.pc_ext:
            self.copy_pext_to_h()
    
    
    def copy_xvec_to_seq(self, xVec, sVec):
        scatter, tVec = PETSc.Scatter.toAll(xVec)

        scatter.begin(xVec, tVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (xVec, tVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        
        sVec.getArray()[:] = tVec.getValues(range(0, self.grid.nx)).copy()
#         tVec.copy(sVec)
        
        scatter.destroy()
        tVec.destroy()
    
    
    def copy_pint_to_h(self):
        self.toolbox.potential_to_hamiltonian(self.pc_int, self.h1c)
        

    def copy_pext_to_h(self):
        self.toolbox.potential_to_hamiltonian(self.pc_ext, self.h2c)
        
        
    def make_history(self):
        self.fc.copy(self.fh)
        self.h1c.copy(self.h1h)
        self.h2c.copy(self.h2h)
        
        self.pc_int.copy(self.ph_int)
        self.pc_ext.copy(self.ph_ext)
        
        self.nc.copy(self.nh)
        self.uc.copy(self.uh)
        self.ec.copy(self.eh)
        self.ac.copy(self.ah)
        
        if self.vlasov_solver != None:
            self.vlasov_solver.update_history(self.fc)
        
        if self.cfg['solver']['initial_guess'] == "gear":
            self.arakawa_gear.update_history(self.fc)
        
    
    def save_to_hdf5(self, itime):
        # save to hdf5 file
        if itime % self.nsave == 0 or itime == self.nt + 1:
            self.hdf5_viewer.incrementTimestep(1)
            self.save_hdf5_vectors()


    def save_hdf5_vectors(self):
        self.hdf5_viewer(self.time)
        self.hdf5_viewer(self.h0)
        self.hdf5_viewer(self.h1c)
        self.hdf5_viewer(self.h2c)
        self.hdf5_viewer(self.fc)
        self.hdf5_viewer(self.pc_int)
        self.hdf5_viewer(self.pc_ext)
        self.hdf5_viewer(self.N)
        self.hdf5_viewer(self.U)
        self.hdf5_viewer(self.E)

    
    def initial_guess(self):
        # backup previous step
        self.fc.copy(self.fl)
        
        # compute norm of previous step
        prev_norm = self.calculate_residual()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Previous Step:                             residual = %24.16E" % (prev_norm))
        
        # calculate initial guess
        self.initial_guess_method()
        
        # check if residual went down
        ig_norm = self.calculate_residual()
        
        # if residual of previous step is smaller then initial guess
        # copy back previous step
        if ig_norm > prev_norm:
            self.fl.copy(self.fc)
    
    
    def initial_guess_none(self):
        pass
        
    
    def initial_guess_rk4(self):
        """
        Calculate initial guess for distribution function
        using Runge-Kutta-4 timestepping together with
        Arakawa's 4th order bracket discretisation.
        """

        for i in range(0, self.nInitial):
#             self.arakawa_rk4.rk4_J1(self.fc)
#             self.arakawa_rk4.rk4_J2(self.fc)
#             self.arakawa_rk4.rk4_J4(self.fc)
            
            self.arakawa_rk4.rk18_J4(self.fc)
            self.calculate_moments(output=False)
        
        ignorm = self.calculate_residual()
        phisum = self.pc_int.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  RK4 Initial Guess:                         residual = %24.16E" % (ignorm))
            print("                                             sum(phi) = %24.16E" % (phisum))
         
    
    def initial_guess_gear(self):
        """
        Calculate initial guess for distribution function
        using 4th order Gear timestepping together with
        Arakawa's 4th order bracket discretisation.
        """
        
        for i in range(0, self.nInitial):
#             self.arakawa_gear.gear2(self.fc)
#             self.arakawa_gear.gear3(self.fc)
            self.arakawa_gear.gear4(self.fc)
            self.calculate_moments(output=False)
            
            if i < self.nInitial-1:
                self.arakawa_gear.update_history(self.fc)
        
        ignorm = self.calculate_residual()
        phisum = self.pc_int.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Gear Initial Guess:                        residual = %24.16E" % (ignorm))
            print("                                             sum(phi) = %24.16E" % (phisum))
         
        
    def initial_guess_symplectic2(self):
        """
        Calculate initial guess for distribution function
        using 2nd order symplectic timestepping together with
        Arakawa's 4th order bracket discretisation.
        """

        for i in range(0, self.nInitial):
            
            self.arakawa_symplectic.kinetic(self.fc, 0.5)
#             self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.fc, 1.0)

            self.arakawa_symplectic.kinetic(self.fc, 0.5)
            self.calculate_moments(output=False)
            
        ignorm = self.calculate_residual()
        phisum = self.pc_int.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Symplectic Initial Guess:                  residual = %24.16E" % (ignorm))
            print("                                             sum(phi) = %24.16E" % (phisum))
         
    
    def initial_guess_symplectic4(self):
        """
        Calculate initial guess for distribution function
        using 4th order symplectic timestepping together with
        Arakawa's 4th order bracket discretisation.
        """
        
        fac2 = 2.**(1./3.)
        
        c1 = 0.5 / ( 2. - fac2 )
        c2 = c1  * ( 1. - fac2 )
        c3 = c2
        c4 = c1
        
        d1 = 1. / ( 2. - fac2 )
        d2 = - d1 * fac2
        d3 = d1
        
        for i in range(0, self.nInitial):
            
            self.arakawa_symplectic.kinetic(self.fc, c1)
#             self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.fc, d1)
            
            self.arakawa_symplectic.kinetic(self.fc, c2)
#             self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.fc, d2)
            
            self.arakawa_symplectic.kinetic(self.fc, c3)
#             self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.fc, d3)
            
            self.arakawa_symplectic.kinetic(self.fc, c4)
            self.calculate_moments(output=False)
            
        
        ignorm = self.calculate_residual()
        phisum = self.pc_int.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Symplectic Initial Guess:                  residual = %24.16E" % (ignorm))
            print("                                             sum(phi) = %24.16E" % (phisum))
         
    
