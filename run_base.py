'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys
import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

from vlasov.VIDA    import VIDA
from vlasov.Toolbox import Toolbox

from vlasov.core.config  import Config
from vlasov.data.maxwell import maxwellian

from vlasov.predictor.PETScArakawaRK4       import PETScArakawaRK4
from vlasov.predictor.PETScArakawaGear      import PETScArakawaGear
from vlasov.predictor.PETScPoissonMatrixJ4  import PETScPoissonMatrix


class petscVP1Dbase():
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
        # number of iterations for initial guess
        self.nInitial = 1
#         self.nInitial = 4
#         self.nInitial = 10
#         self.nInitial = 100
        
        
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
        vMin    = self.cfg['grid']['vmin']
        vMax    = self.cfg['grid']['vmax']
        
        self.hx = L / self.nx                       # gridstep size in x
        self.hv = (vMax - vMin) / (self.nv-1)       # gridstep size in v
        
        self.time = PETSc.Vec().createMPI(1, 1, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        self.coll_freq = self.cfg['solver']['coll_freq']             # collision frequency
        
        self.charge = self.cfg['initial_data']['charge']             # particle charge
        self.mass   = self.cfg['initial_data']['mass']               # particle mass
        
        
        hdf_out_filename = self.cfg['io']['hdf5_output']
        cfg_out_filename = hdf_out_filename.replace('.hdf5', '.cfg') 
        
        self.cfg.write_current_config(cfg_out_filename)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_rtol',   self.cfg['solver']['petsc_ksp_rtol'])
        OptDB.setValue('ksp_atol',   self.cfg['solver']['petsc_ksp_atol'])
        OptDB.setValue('ksp_max_it', self.cfg['solver']['petsc_ksp_max_iter'])

        OptDB.setValue('snes_rtol',   self.cfg['solver']['petsc_snes_rtol'])
        OptDB.setValue('snes_atol',   self.cfg['solver']['petsc_snes_atol'])
        OptDB.setValue('snes_stol',   self.cfg['solver']['petsc_snes_stol'])
        OptDB.setValue('snes_max_it', self.cfg['solver']['petsc_snes_max_iter'])
        
        
        # create DA for 2d grid (f only)
        self.da1 = VIDA().create(dim=1, dof=self.nv,
                                       sizes=[self.nx],
                                       proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                       boundary_type=('periodic'),
                                       stencil_width=2,
                                       stencil_type='box')
        
        # create VIDA for 2d grid (f, phi and moments)
        self.da2 = VIDA().create(dim=1, dof=self.nv+4,
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
        self.da2.setUniformCoordinates(xmin=0.0,  xmax=L)
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
        
        
        # create solution and RHS vector
        self.x  = self.da2.createGlobalVec()
        self.xh = self.da2.createGlobalVec()
        self.b  = self.da2.createGlobalVec()
        self.x_nvec = self.da2.createGlobalVec()
        
        # create solution and RHS vector for Vlasov and Poisson solver
        self.pb = self.dax.createGlobalVec()
        self.p_nvec = self.dax.createGlobalVec()
        
        # create vectors for
        # Hamiltonians
        self.h0 = self.da1.createGlobalVec()
        self.h1 = self.da1.createGlobalVec()
        self.h2 = self.da1.createGlobalVec()
        
        # distribution functions
        self.f     = self.da1.createGlobalVec()
        
        # moments
        self.n     = self.dax.createGlobalVec()
        self.u     = self.dax.createGlobalVec()
        self.e     = self.dax.createGlobalVec()
        
        # potential
        self.p     = self.dax.createGlobalVec()
        self.p_ext = self.dax.createGlobalVec()
        
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
        self.p_nvec.set(1.)
        self.p_nvec.normalize()
        
        self.x_nvec.set(0.)
        x_nvec_arr = self.da2.getGlobalArray(self.x_nvec)
        p_nvec_arr = self.dax.getGlobalArray(self.p_nvec)
        
        x_nvec_arr[:, self.nv] = p_nvec_arr  
#         x_nvec_arr[:, self.nv] = 1.
#         self.x_nvec.normalize()
        
        self.p_nullspace = PETSc.NullSpace().create(constant=False, vectors=(self.p_nvec,))
        self.nullspace = PETSc.NullSpace().create(constant=False, vectors=(self.x_nvec,))
        
        
        # initialise kinetic hamiltonian
        h0_arr = self.da1.getVecArray(self.h0)
        
        for i in range(xs, xe):
            for j in range(0, self.nv):
                h0_arr[i, j] = 0.5 * self.vGrid[j]**2 * self.mass
        
        
        # create Arakawa RK4 solver object
        self.arakawa_rk4 = PETScArakawaRK4(self.da1, self.da2, self.dax,
                                           self.h0, self.vGrid,
                                           self.nx, self.nv, self.ht / float(self.nInitial), self.hx, self.hv)
        
        self.arakawa_gear = PETScArakawaGear(self.da1, self.da2, self.dax,
                                             self.h0, self.vGrid,
                                             self.nx, self.nv, self.ht / float(self.nInitial), self.hx, self.hv)
        
        # create Toolbox
        self.toolbox = Toolbox(self.da1, self.da2, self.dax, self.vGrid, self.nx, self.nv, self.ht, self.hx, self.hv)
        
        
        # initialise matrix
        self.A = self.da2.createMat()
        self.A.setOption(self.A.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.A.setUp()
        self.A.setNullSpace(self.nullspace)

        # initialise Jacobian
        self.J = self.da2.createMat()
        self.J.setOption(self.J.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.J.setUp()
        self.J.setNullSpace(self.nullspace)


        # create placeholder for solver object
        self.petsc_solver = None
        
        # create Poisson object
        self.poisson_mat = PETScPoissonMatrix(self.da1, self.dax, 
                                              self.nx, self.nv, self.hx, self.hv,
                                              self.charge)
        
        # initialise Poisson matrix
        self.poisson_A = self.dax.createMat()
        self.poisson_A.setOption(self.poisson_A.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.poisson_A.setUp()
        self.poisson_A.setNullSpace(self.p_nullspace)
        
        # create linear Poisson solver
        OptDB.setValue('ksp_rtol', 1E-13)

        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_A)
        self.poisson_ksp.setType('cg')
#         self.poisson_ksp.setType('bcgs')
        self.poisson_ksp.getPC().setType('none')
        
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
        
        n0_arr = self.dax.getVecArray(n0)
        T0_arr = self.dax.getVecArray(T0)
        f_arr  = self.da1.getVecArray(self.f)
        
        
        if self.cfg['initial_data']['distribution_python'] != None:
            init_data = __import__("runs." + self.cfg['initial_data']['distribution_python'], globals(), locals(), ['distribution'], 0)
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("Initialising distribution function with Python function.")
            
            for i in range(xs, xe):
                for j in range(0, self.nv):
                    if j <= 1 or j >= self.nv-2:
                        f_arr[i,j] = 0.0
                    else:
                        f_arr[i,j] = init_data.distribution(self.xGrid[i], self.vGrid[j]) 
            
            n0_arr[xs:xe] = 0.
            T0_arr[xs:xe] = 0.
        
        else:
            if self.cfg['initial_data']['density_python'] != None:
                init_data = __import__("runs." + self.cfg['initial_data']['density_python'], globals(), locals(), ['density'], 0)
                
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("Initialising density with Python function.")
            
                for i in range(xs, xe):
                    n0_arr[i] = init_data.density(self.xGrid[i], L) 
            
            else:
                n0_arr[xs:xe] = self.cfg['initial_data']['density']            
            
            
            if self.cfg['initial_data']['temperature_python'] != None:
                init_data = __import__("runs." + self.cfg['initial_data']['temperature_python'], globals(), locals(), ['temperature'], 0)
                
                if PETSc.COMM_WORLD.getRank() == 0:
                    print("Initialising temperature with Python function.")
            
                for i in range(xs, xe):
                    T0_arr[i] = init_data.temperature(self.xGrid[i]) 
            
            else:
                T0_arr[xs:xe] = self.cfg['initial_data']['temperature']            
            
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("Initialising distribution function with Maxwellian.")
            
            for i in range(xs, xe):
                for j in range(0, self.nv):
                    if j <= 1 or j >= self.nv-2:
                        f_arr[i,j] = 0.0
                    else:
                        f_arr[i,j] = n0_arr[i] * maxwellian(T0_arr[i], self.vGrid[j])
        
        
        # normalise f to fit density and copy f to x
        if PETSc.COMM_WORLD.getRank() == 0:
            print("Normalise distribution function.")
        nave = self.f.sum() * self.hv / self.nx
        self.f.scale(1./nave)
        self.copy_f_to_x()
        
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
        
        # update solution history
        self.arakawa_gear.update_history(self.f, self.h1)
        
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
    
    
    
    def calculate_moments(self, potential=True, output=True):
        self.toolbox.compute_density(self.f, self.n)
        self.toolbox.compute_velocity_density(self.f, self.u)
        self.toolbox.compute_energy_density(self.f, self.e)
 
        self.copy_n_to_x()                    # copy density to solution vector
        self.copy_u_to_x()                    # copy velocity to solution vector
        self.copy_e_to_x()                    # copy energy to solution vector
        
        if potential:
            self.calculate_potential(output)      # calculate initial potential
            self.copy_p_to_x()                    # copy potential to solution vector
            self.copy_p_to_h()                    # copy potential to Hamiltonian
    
    
    def calculate_potential(self, output=True):
        
        self.poisson_mat.formMat(self.poisson_A)
        self.poisson_mat.formRHS(self.n, self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        if output:
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Poisson:                            sum(phi) = %24.16E" % (phisum))
    
    
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
    
    
    def copy_x_to_data(self):
        self.copy_x_to_f()
        self.copy_x_to_p()
        self.copy_x_to_n()
        self.copy_x_to_u()
        self.copy_x_to_e()
        self.copy_p_to_h()
    
    
    def copy_data_to_x(self):
        self.copy_f_to_x()
        self.copy_p_to_x()
        self.copy_n_to_x()
        self.copy_u_to_x()
        self.copy_e_to_x()
    
    
    def copy_x_to_f(self):
        x_arr = self.da2.getGlobalArray(self.x)
        f_arr = self.da1.getGlobalArray(self.f)
        
        f_arr[:, :] = x_arr[:, 0:self.nv] 
        
    
    def copy_f_to_x(self):
        x_arr = self.da2.getGlobalArray(self.x)
        f_arr = self.da1.getGlobalArray(self.f)
        
        x_arr[:, 0:self.nv] = f_arr[:, :]
        
    
    def copy_x_to_p(self):
        x_arr = self.da2.getGlobalArray(self.x)
        p_arr = self.dax.getGlobalArray(self.p)
        
        p_arr[:] = x_arr[:, self.nv]
        
    
    def copy_p_to_x(self):
        p_arr = self.dax.getGlobalArray(self.p)
        x_arr = self.da2.getGlobalArray(self.x)
        
        x_arr[:, self.nv] = p_arr[:]
        
        
    def copy_p_to_h(self):
        self.toolbox.potential_to_hamiltonian(self.p, self.h1)
        

    def copy_pext_to_h(self):
        self.toolbox.potential_to_hamiltonian(self.p_ext, self.h2)
        

    def copy_x_to_n(self):
        x_arr = self.da2.getGlobalArray(self.x)
        n_arr = self.dax.getGlobalArray(self.n)
        
        n_arr[:] = x_arr[:, self.nv+1]
        
    
    def copy_n_to_x(self):
        n_arr = self.dax.getGlobalArray(self.n)
        x_arr = self.da2.getGlobalArray(self.x)
        
        x_arr[:, self.nv+1] = n_arr[:]
        
        
    def copy_x_to_u(self):
        x_arr = self.da2.getGlobalArray(self.x)
        u_arr = self.dax.getGlobalArray(self.u)
        
        u_arr[:] = x_arr[:, self.nv+2]
        
    
    def copy_u_to_x(self):
        u_arr = self.dax.getGlobalArray(self.u)
        x_arr = self.da2.getGlobalArray(self.x)
        
        x_arr[:, self.nv+2] = u_arr[:]
        
        
    def copy_x_to_e(self):
        x_arr = self.da2.getGlobalArray(self.x)
        e_arr = self.dax.getGlobalArray(self.e)
        
        e_arr[:] = x_arr[:, self.nv+3]
        
    
    def copy_e_to_x(self):
        e_arr = self.dax.getGlobalArray(self.e)
        x_arr = self.da2.getGlobalArray(self.x)
        
        x_arr[:, self.nv+3] = e_arr[:]
        
        
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

    
    def initial_guess_rk4(self):
        # calculate initial guess for distribution function

        for i in range(0, self.nInitial):
#             self.arakawa_rk4.rk4_J1(self.f, self.h1)
#             self.arakawa_rk4.rk4_J2(self.f, self.h1)
            self.arakawa_rk4.rk4_J4(self.f, self.h1)
            
            self.copy_f_to_x()
            
            self.calculate_moments(output=False)
        
        
        self.petsc_solver.function_mult(self.x, self.b)
        ignorm = self.b.norm()
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  RK4 Initial Guess:                      funcnorm = %24.16E" % (ignorm))
            print("                                          sum(phi) = %24.16E" % (phisum))
         
    
    def initial_guess_gear(self, itime):
        if itime == 1:
            self.initial_guess_rk4()
        
        else:
            if itime == 2:
                gear = self.arakawa_gear.gear2
            elif itime == 3:
                gear = self.arakawa_gear.gear3
            elif itime >= 4:
                gear = self.arakawa_gear.gear4
                
            
            for i in range(0, self.nInitial):
                gear(self.f)
                
                self.copy_f_to_x()
                
                self.calculate_moments(output=False)
                
                if i < self.nInitial-1:
                    self.arakawa_gear.update_history(self.f, self.h1)
            
            
            self.petsc_solver.function_mult(self.x, self.b)
            ignorm = self.b.norm()
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Gear Initial Guess:                     funcnorm = %24.16E" % (ignorm))
                print("                                          sum(phi) = %24.16E" % (phisum))
         
        
        
    
