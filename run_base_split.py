'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, time, datetime

import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

from vlasov.core.config  import Config

from vlasov.toolbox.VIDA    import VIDA
# from vlasov.toolbox.Arakawa import Arakawa
from vlasov.toolbox.Toolbox import Toolbox

from vlasov.explicit.PETScArakawaRK4        import PETScArakawaRK4
from vlasov.explicit.PETScArakawaGear       import PETScArakawaGear
from vlasov.explicit.PETScArakawaSymplectic import PETScArakawaSymplectic

from vlasov.solvers.poisson.PETScPoissonSolver2  import PETScPoissonSolver


class petscVP1Dbasesplit():
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
        
        
        # load run config file
        self.cfg = Config(cfgfile)
        
        # timestep setup
        self.ht    = self.cfg['grid']['ht']              # timestep size
        self.nt    = self.cfg['grid']['nt']              # number of timesteps
        self.nsave = self.cfg['io']['nsave']             # save only every nsave'th timestep
        
        # grid setup
        self.nx = self.cfg['grid']['nx']                 # number of points in x
        self.nv = self.cfg['grid']['nv']                 # number of points in v
        L       = self.cfg['grid']['L']
        vMax    = self.cfg['grid']['vmax']
        vMin    = -vMax
        
        self.Lx = L
        self.Lv = vMax - vMin
        
        self.hx = self.Lx / self.nx                      # gridstep size in x
        self.hv = self.Lv / (self.nv-1)                  # gridstep size in v
        
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
        self.da1 = VIDA().create(dim=1, dof=self.nv,
                                       sizes=[self.nx],
                                       proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                       boundary_type=('periodic'),
                                       stencil_width=2,
                                       stencil_type='box')
        
        # create VIDA for x grid
        self.dax = VIDA().create(dim=1, dof=1,
                                       sizes=[self.nx],
                                       proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                       boundary_type=('periodic'),
                                       stencil_width=2,
                                       stencil_type='box')
        
        # create VIDA for y grid
        self.day = VIDA().create(dim=1, dof=1,
                                       sizes=[self.nv],
                                       proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                       boundary_type=('none'))
        
        
        # initialise grid
        self.da1.setUniformCoordinates(xmin=0.0,  xmax=L)
        self.dax.setUniformCoordinates(xmin=0.0,  xmax=L)
        self.day.setUniformCoordinates(xmin=vMin, xmax=vMax) 
        
        # get local index ranges
        (xs, xe), = self.da1.getRanges()
        
        # get coordinate vectors
        coords_x = self.dax.getCoordinates()
        coords_v = self.day.getCoordinates()
        
        # save x coordinate arrays
        scatter, xVec = PETSc.Scatter.toAll(coords_x)

        scatter.begin(coords_x, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_x, xVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                  
        self.xGrid = xVec.getValues(range(0, self.nx)).copy()
        
        scatter.destroy()
        xVec.destroy()
        
        # save v coordinate arrays
        scatter, vVec = PETSc.Scatter.toAll(coords_v)

        scatter.begin(coords_v, vVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        scatter.end  (coords_v, vVec, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
                  
        self.vGrid = vVec.getValues(range(0, self.nv)).copy()
        
        scatter.destroy()
        vVec.destroy()
        
        
        # create vectors for Hamiltonians
        self.h0 = self.da1.createGlobalVec()        # kinetic Hamiltonian
        self.h1 = self.da1.createGlobalVec()        # potential Hamiltonian
        self.h2 = self.da1.createGlobalVec()        # external Hamiltonian
        
        # distribution functions
        self.f     = self.da1.createGlobalVec()     # distribution function
        self.fh    = self.da1.createGlobalVec()     # history
        self.fb    = self.da1.createGlobalVec()     # right hand side
        self.df     = self.da1.createGlobalVec()    # delta
        
        # moments
        self.n     = self.dax.createGlobalVec()     # density
        self.u     = self.dax.createGlobalVec()     # velocity density
        self.e     = self.dax.createGlobalVec()     # energy density
        
        # potential
        self.p     = self.dax.createGlobalVec()     # potential
        self.pb    = self.dax.createGlobalVec()     # right hand side
        self.pn    = self.dax.createGlobalVec()     # null vector
        self.p_ext = self.dax.createGlobalVec()     # external potential
        
        # set variable names
        self.h0.setName('h0')
        self.h1.setName('h1')
        self.h2.setName('h2')
        self.f.setName('f')
        self.n.setName('n')
        self.u.setName('u')
        self.e.setName('e')
        self.p.setName('phi')
        self.p_ext.setName('phi_ext')
        
        
        # initialise nullspace basis vectors
        # the Poisson equation has a null space of all constant vectors
        # that needs to be removed to avoid jumpy potentials
        self.pn.set(1.)
        self.pn.normalize()
        
        self.p_nullspace = PETSc.NullSpace().create(constant=False, vectors=(self.pn,))
        
        
        # create Toolbox
#         self.arakawa = Arakawa(self.da1, self.dax, self.vGrid, self.nx, self.nv, self.ht, self.hx, self.hv)
        self.toolbox = Toolbox(self.da1, self.dax, self.vGrid, self.nx, self.nv, self.ht, self.hx, self.hv)
        
        
        # initialise kinetic hamiltonian
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Initialising kinetic Hamiltonian.")
        self.toolbox.initialise_kinetic_hamiltonian(self.h0, self.mass)
                
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Instantiating Initial Guess Objects.")
            
        # create Arakawa RK4 solver object
        self.arakawa_rk4 = PETScArakawaRK4(self.da1, self.dax,
                                           self.h0, self.vGrid,
                                           self.nx, self.nv, self.ht / float(self.nInitial), self.hx, self.hv)
        
        self.arakawa_gear = PETScArakawaGear(self.da1, self.dax,
                                             self.h0, self.vGrid,
                                             self.nx, self.nv, self.ht / float(self.nInitial), self.hx, self.hv)
        
        self.arakawa_symplectic = PETScArakawaSymplectic(self.da1, self.dax,
                                                         self.h0, self.vGrid,
                                                         self.nx, self.nv, self.ht / float(self.nInitial), self.hx, self.hv)
        

        if PETSc.COMM_WORLD.getRank() == 0:
            print("Instantiating Poisson Object.")
            
        # initialise Poisson matrix
        self.poisson_A = self.dax.createMat()
#         self.poisson_A.setOption(self.poisson_A.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.poisson_A.setUp()
        self.poisson_A.setNullSpace(self.p_nullspace)
        
        # create Poisson object
        self.poisson_solver = PETScPoissonSolver(self.dax, self.nx, self.hx, self.charge)
        self.poisson_solver.formMat(self.poisson_A)
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Create Poisson Solver.")
            
        # create linear Poisson solver
        OptDB.setValue('ksp_rtol', 1E-13)

#         OptDB.setValue('pc_gamg_type', 'agg')
#         OptDB.setValue('pc_gamg_agg_nsmooths', '1')
        
#         OptDB.setValue('pc_hypre_type', 'boomeramg')


        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_A)
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
            print("Config File: %s" % cfgfile)
            print("Output File: %s" % hdf_out_filename)
            print()
            print("nt = %i" % (self.nt))
            print("nx = %i" % (self.nx))
            print("nv = %i" % (self.nv))
            print()
            print("ht = %e" % (self.ht))
            print("hx = %e" % (self.hx))
            print("hv = %e" % (self.hv))
            print()
            print("Lx   =  %12.6e" % (L))
            print("vMin = %+12.6e" % (vMin))
            print("vMax = %+12.6e" % (vMax))
            print()
            print("nu   = %7.1e" % (self.coll_freq))
            print()
            print("CFL  = %e" % (self.hx / vMax))
            print()
            print()
        
        
        # set initial data
        n0 = self.dax.createGlobalVec()
        T0 = self.dax.createGlobalVec()
        
        n0.setName('n0')
        T0.setName('T0')
        
        n0_arr = self.dax.getGlobalArray(n0)
        T0_arr = self.dax.getGlobalArray(T0)
        f_arr  = self.da1.getGlobalArray(self.f)
        x_arr  = self.dax.getGlobalArray(coords_x)
        
        
        if self.cfg['initial_data']['distribution_python'] != None:
            init_data = __import__("runs." + self.cfg['initial_data']['distribution_python'], globals(), locals(), ['distribution'], 0)
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("Initialising distribution function with Python function.")
            
            self.toolbox.initialise_distribution_function(f_arr, x_arr, init_data.distribution)
            
#             for i in range(0, xe-xs):
#                 for j in range(0, self.nv):
#                     if j <= 1 or j >= self.nv-2:
#                         f_arr[i,j] = 0.0
#                     else:
#                         f_arr[i,j] = init_data.distribution(self.xGrid[i], self.vGrid[j]) 
            
            n0_arr[:] = 0.
            T0_arr[:] = 0.
        
        else:
            if self.cfg['initial_data']['density_python'] != None:
                init_data = __import__("runs." + self.cfg['initial_data']['density_python'], globals(), locals(), ['density'], 0)
                
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("Initialising density with Python function.")
            
                for i in range(0, xe-xs):
                    n0_arr[i] = init_data.density(x_arr[i], L) 
            
            else:
                n0_arr[:] = self.cfg['initial_data']['density']            
            
            
            if self.cfg['initial_data']['temperature_python'] != None:
                init_data = __import__("runs." + self.cfg['initial_data']['temperature_python'], globals(), locals(), ['temperature'], 0)
                
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("Initialising temperature with Python function.")
            
                for i in range(0, xe-xs):
                    T0_arr[i] = init_data.temperature(x_arr[i]) 
            
            else:
                T0_arr[:] = self.cfg['initial_data']['temperature']            
            
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("Initialising distribution function with Maxwellian.")
            
            self.toolbox.initialise_distribution_nT(f_arr, n0_arr, T0_arr)
            
#             for i in range(0, xe-xs):
#                 for j in range(0, self.nv):
#                     if j <= 1 or j >= self.nv-2:
#                         f_arr[i,j] = 0.0
#                     else:
#                         f_arr[i,j] = n0_arr[i] * maxwellian(T0_arr[i], self.vGrid[j])
        
        
        # normalise f
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Normalise distribution function.")
        nave = self.f.sum() * self.hv / self.nx
        self.f.scale(1./nave)
        
        
        # calculate potential and moments
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Calculate initial potential and moments.")
        self.calculate_moments(output=False)
        
        
        # check for external potential
        if self.cfg['initial_data']['external_python'] != None:
            if PETSc.COMM_WORLD.getRank() == 0:
                print("Calculate external potential.")
                
            external_data = __import__("runs." + self.cfg['initial_data']['external_python'], globals(), locals(), ['external'], 0)
            self.external = external_data.external
        else:
            self.external = None
        
        self.p_ext.set(0.)
        self.calculate_external(0.)
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Copy initial data to matrix and function objects.")
        
        # copy external potential
        ### TODO ###
        # implement update_external() in ArakawaGear and ArakawaRK4 Module
        ### TODO ###
#         self.arakawa_rk4.update_external(self.p_ext)
#         self.arakawa_gear.update_external(self.p_ext)
        
        
        # update gear solution history
        gear_arakawa_rk4 = PETScArakawaRK4(self.da1, self.dax,
                                           self.h0, self.vGrid,
                                           self.nx, self.nv, - self.ht / float(self.nInitial), self.hx, self.hv)
        
        f_gear = []
        h_gear = []
        
        f_gear.append(self.da1.createGlobalVec())
        h_gear.append(self.da1.createGlobalVec())
        
        self.f.copy(f_gear[0])
        self.h1.copy(h_gear[0])
        
        for i in range(1,4):
            f_gear.append(self.da1.createGlobalVec())
            h_gear.append(self.da1.createGlobalVec())
            
            f_gear[i-1].copy(f_gear[i])
            h_gear[i-1].copy(h_gear[i])
            
            gear_arakawa_rk4.rk4_J4(f_gear[i], h_gear[i])
            
            self.calculate_moments(output=False)
            self.h1.copy(h_gear[i])
            
        
        for i in range(0,4):
            self.arakawa_gear.update_history(f_gear[3-i], h_gear[3-i])
        
        del gear_arakawa_rk4
        del f_gear
        del h_gear
        
        # recover correct moments and potential
        self.calculate_moments(output=False)
        
        
        # create HDF5 output file
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Create HDF5 output file.")
        
        self.hdf5_viewer = PETSc.ViewerHDF5().create(hdf_out_filename,
                                              mode=PETSc.Viewer.Mode.WRITE,
                                              comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.pushGroup("/")
        
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
            print("")
    
    
    def calculate_residual(self):
        self.vlasov_solver.function_mult(self.f, self.fb)
        fnorm = self.fb.norm()
        
        self.poisson_solver.function_mult(self.p, self.n, self.pb)
        pnorm = self.pb.norm()
        
        return fnorm + pnorm
    
    
    def calculate_moments(self, potential=True, output=True):
        self.toolbox.compute_density(self.f, self.n)
        self.toolbox.compute_velocity_density(self.f, self.u)
        self.toolbox.compute_energy_density(self.f, self.e)
 
        if potential:
            self.calculate_potential(output)      # calculate initial potential
            self.copy_p_to_h()                    # copy potential to Hamiltonian
    
    
    def calculate_potential(self, output=True):
        
        self.poisson_solver.formRHS(self.n, self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        if output:
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Poisson:                               sum(phi) = %24.16E" % (phisum))
    
    
    def calculate_external(self, t):
        (xs, xe), = self.da1.getRanges()
        
        if self.external != None:
            p_ext_arr = self.dax.getVecArray(self.p_ext)
            
            for i in range(xs, xe):
                p_ext_arr[i] = self.external(self.xGrid[i], t) 
            
            # remove average
            phisum = self.p_ext.sum()
            phiave = phisum / self.nx
            self.p_ext.shift(-phiave)
    
        self.copy_pext_to_h()
    
    
    def copy_p_to_h(self):
        self.toolbox.potential_to_hamiltonian(self.p, self.h1)
        

    def copy_pext_to_h(self):
        self.toolbox.potential_to_hamiltonian(self.p_ext, self.h2)
        

    def save_to_hdf5(self, itime):
        # save to hdf5 file
        if itime % self.nsave == 0 or itime == self.nt + 1:
            self.hdf5_viewer.incrementTimestep(1)
            self.save_hdf5_vectors()


    def save_hdf5_vectors(self):
        self.hdf5_viewer(self.time)
        self.hdf5_viewer(self.f)
        self.hdf5_viewer(self.n)
        self.hdf5_viewer(self.u)
        self.hdf5_viewer(self.e)
        self.hdf5_viewer(self.p)
        self.hdf5_viewer(self.p_ext)
        self.hdf5_viewer(self.h0)
        self.hdf5_viewer(self.h1)
        self.hdf5_viewer(self.h2)

    
    def initial_guess(self):
        # backup previous step
        self.f.copy(self.fh)
        
        # compute norm of previous step
        prev_norm = self.calculate_residual()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Previous Step:                             funcnorm = %24.16E" % (prev_norm))
        
        # calculate initial guess
        self.initial_guess_method()
        
        # check if residual went down
        ig_norm = self.calculate_residual()
        
        # if residual of previous step is smaller then initial guess
        # copy back previous step
        if ig_norm > prev_norm:
            self.fh.copy(self.f)
    
    
    def initial_guess_none(self):
        pass
        
    
    def initial_guess_rk4(self):
        """
        Calculate initial guess for distribution function
        using Runge-Kutta-4 timestepping together with
        Arakawa's 4th order bracket discretisation.
        """

        for i in range(0, self.nInitial):
#             self.arakawa_rk4.rk4_J1(self.f, self.h1)
#             self.arakawa_rk4.rk4_J2(self.f, self.h1)
            self.arakawa_rk4.rk4_J4(self.f, self.h1)
            self.calculate_moments(output=False)
        
        ignorm = self.calculate_residual()
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  RK4 Initial Guess:                         funcnorm = %24.16E" % (ignorm))
            print("                                             sum(phi) = %24.16E" % (phisum))
         
    
    def initial_guess_gear(self):
        """
        Calculate initial guess for distribution function
        using 4th order Gear timestepping together with
        Arakawa's 4th order bracket discretisation.
        """
        
        for i in range(0, self.nInitial):
#             self.arakawa_gear.gear2(self.f)
#             self.arakawa_gear.gear3(self.f)
            self.arakawa_gear.gear4(self.f)
            self.calculate_moments(output=False)
            
            if i < self.nInitial-1:
                self.arakawa_gear.update_history(self.f, self.h1)
        
        ignorm = self.calculate_residual()
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Gear Initial Guess:                        funcnorm = %24.16E" % (ignorm))
            print("                                             sum(phi) = %24.16E" % (phisum))
         
        
    def initial_guess_symplectic2(self):
        """
        Calculate initial guess for distribution function
        using 2nd order symplectic timestepping together with
        Arakawa's 4th order bracket discretisation.
        """

        for i in range(0, self.nInitial):
            
            self.arakawa_symplectic.kinetic(self.f, 0.5)
            self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.f, self.h1, 1.0)

            self.arakawa_symplectic.kinetic(self.f, 0.5)
            self.calculate_moments(output=False)
            
        ignorm = self.calculate_residual()
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Symplectic Initial Guess:                  funcnorm = %24.16E" % (ignorm))
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
            
            self.arakawa_symplectic.kinetic(self.f, c1)
            self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.f, self.h1, d1)
            
            self.arakawa_symplectic.kinetic(self.f, c2)
            self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.f, self.h1, d2)
            
            self.arakawa_symplectic.kinetic(self.f, c3)
            self.calculate_moments(output=False)
            self.arakawa_symplectic.potential(self.f, self.h1, d3)
            
            self.arakawa_symplectic.kinetic(self.f, c4)
            self.calculate_moments(output=False)
            
            
        ignorm = self.calculate_residual()
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Symplectic Initial Guess:                  funcnorm = %24.16E" % (ignorm))
            print("                                             sum(phi) = %24.16E" % (phisum))
         
    
