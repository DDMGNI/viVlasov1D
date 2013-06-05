'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse
import os, sys, time

import numpy as np

import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

from vlasov.Toolbox import Toolbox

from vlasov.core.config  import Config
from vlasov.data.maxwell import maxwellian

from vlasov.predictor.PETScArakawaRK4       import PETScArakawaRK4
from vlasov.predictor.PETScArakawaGear      import PETScArakawaGear

# from vlasov.vi.PETScMatrixJ1                import PETScMatrix
# from vlasov.vi.PETScFunctionJ1              import PETScFunction
# from vlasov.vi.PETScJacobianJ1              import PETScJacobian
# from vlasov.predictor.PETScPoissonMatrixJ1  import PETScPoissonMatrix

# from vlasov.vi.PETScMatrixJ1                import PETScMatrix
# from vlasov.vi.PETScNLFunctionJ1            import PETScFunction
# from vlasov.vi.PETScNLJacobianJ1            import PETScJacobian
# from vlasov.predictor.PETScPoissonMatrixJ1  import PETScPoissonMatrix

# from vlasov.vi.PETScMatrixJ2                import PETScMatrix
# from vlasov.vi.PETScNLFunctionJ2            import PETScFunction
# from vlasov.vi.PETScNLJacobianJ2            import PETScJacobian
# from vlasov.predictor.PETScPoissonMatrixJ2  import PETScPoissonMatrix

# from vlasov.vi.PETScMatrixJ4                import PETScMatrix
# from vlasov.vi.PETScNLFunctionJ4            import PETScFunction
# from vlasov.vi.PETScNLJacobianJ4            import PETScJacobian
# from vlasov.predictor.PETScPoissonMatrixJ4  import PETScPoissonMatrix

# from vlasov.vi.PETScMatrixJ1                import PETScMatrix
# from vlasov.vi.PETScNLFunctionJ1D4          import PETScFunction
# from vlasov.vi.PETScNLJacobianJ1D4          import PETScJacobian
# from vlasov.predictor.PETScPoissonMatrixJ1  import PETScPoissonMatrix

# from vlasov.vi.PETScMatrixJ1explicit        import PETScMatrix
# from vlasov.vi.PETScNLFunctionJ1explicit    import PETScFunction
# from vlasov.vi.PETScNLJacobianJ1explicit    import PETScJacobian
# from vlasov.predictor.PETScPoissonMatrixJ1  import PETScPoissonMatrix

# from vlasov.vi.PETScMatrixJ1                import PETScMatrix
# # from vlasov.vi.PETScMatrixJ1woa             import PETScMatrix
# from vlasov.vi.PETScNLFunctionJ1woa         import PETScFunction
# from vlasov.vi.PETScNLJacobianJ1woa         import PETScJacobian
# from vlasov.predictor.PETScPoissonMatrixJ1  import PETScPoissonMatrix

from vlasov.vi.PETScMatrixJ4woa                import PETScMatrix
from vlasov.vi.PETScNLFunctionJ4woa            import PETScFunction
from vlasov.vi.PETScNLJacobianJ4woa            import PETScJacobian
from vlasov.vi.PETScNLJacobianMFJ4woa          import PETScJacobianMatrixFree
from vlasov.predictor.PETScPoissonMatrixJ4     import PETScPoissonMatrix


# solver_package = 'superlu_dist'
solver_package = 'mumps'
# solver_package = 'pastix'



class petscVP1D():
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def __init__(self, cfgfile):
        '''
        Constructor
        '''
        
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
        
        self.charge = self.cfg['initial_data']['charge']
        self.mass   = self.cfg['initial_data']['mass']
        
        
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
        
#         OptDB.setValue('snes_lag_preconditioner', 3)
        
#         OptDB.setValue('snes_ls', 'basic')

        OptDB.setValue('ksp_monitor',  '')
        OptDB.setValue('snes_monitor', '')
        
#        OptDB.setValue('log_info',    '')
#        OptDB.setValue('log_summary', '')
        
        
        # create DA for 2d grid (f only)
        self.da1 = PETSc.DA().create(dim=1, dof=self.nv,
                                    sizes=[self.nx],
                                    proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                    boundary_type=('periodic'),
                                    stencil_width=2,
                                    stencil_type='box')
        
        # create DA for 2d grid (f, phi and moments)
        self.da2 = PETSc.DA().create(dim=1, dof=self.nv+6,
                                     sizes=[self.nx],
                                     proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                     boundary_type=('periodic'),
                                     stencil_width=2,
                                     stencil_type='box')
        
        # create DA for x grid
        self.dax = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[self.nx],
                                    proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                    boundary_type=('periodic'),
                                    stencil_width=2,
                                    stencil_type='box')
        
        # create DA for y grid
        self.day = PETSc.DA().create(dim=1, dof=1,
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
        self.nu    = self.dax.createGlobalVec()
        self.ne    = self.dax.createGlobalVec()
        self.u     = self.dax.createGlobalVec()
        self.e     = self.dax.createGlobalVec()
        self.a     = self.dax.createGlobalVec()
        
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
        x_nvec_arr = self.da2.getVecArray(self.x_nvec)[...]
        p_nvec_arr = self.dax.getVecArray(self.p_nvec)[...]
        
        x_nvec_arr[:, self.nv] = 1.  
        self.x_nvec.normalize()
        
        self.nullspace = PETSc.NullSpace().create(constant=False, vectors=(self.x_nvec,))
        
        
        # initialise kinetic hamiltonian
        h0_arr = self.da1.getVecArray(self.h0)
        
        for i in range(xs, xe):
            for j in range(0, self.nv):
                h0_arr[i, j] = 0.5 * self.vGrid[j]**2 # * self.mass
        
        
        # create Jacobian, Function, and linear Matrix objects
        self.petsc_jacobian_mf = PETScJacobianMatrixFree(self.da1, self.da2, self.dax,
                                                         self.h0, self.vGrid,
                                                         self.nx, self.nv, self.ht, self.hx, self.hv,
                                                         self.charge, coll_freq=self.coll_freq)
        
        self.petsc_jacobian = PETScJacobian(self.da1, self.da2, self.dax,
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.charge, coll_freq=self.coll_freq)
        
        self.petsc_function = PETScFunction(self.da1, self.da2, self.dax, 
                                            self.h0, self.vGrid,
                                            self.nx, self.nv, self.ht, self.hx, self.hv,
                                            self.charge, coll_freq=self.coll_freq)
        
        self.petsc_matrix = PETScMatrix(self.da1, self.da2, self.dax,
                                        self.h0, self.vGrid,
                                        self.nx, self.nv, self.ht, self.hx, self.hv,
                                        self.charge)#, coll_freq=self.coll_freq)
        
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

        # initialise matrixfree Jacobian
        self.Jmf = PETSc.Mat().createPython([self.x.getSizes(), self.b.getSizes()], 
                                            context=self.petsc_jacobian_mf,
                                            comm=PETSc.COMM_WORLD)
        self.Jmf.setUp()
        
        
        # create linear solver
        self.snes_linear = PETSc.SNES().create()
        self.snes_linear.setType('ksponly')
        self.snes_linear.setFunction(self.petsc_matrix.snes_mult, self.b)
        self.snes_linear.setJacobian(self.updateMatrix, self.A)
        self.snes_linear.setFromOptions()
        self.snes_linear.getKSP().setType('gmres')
#        self.snes_linear.getKSP().getPC().setType('bjacobi')
        self.snes_linear.getKSP().getPC().setType('asm')
#         self.snes_linear.getKSP().setType('preonly')
#         self.snes_linear.getKSP().getPC().setType('lu')
#         self.snes_linear.getKSP().getPC().setFactorSolverPackage(solver_package)

        # create nonlinear solver
        self.snes = PETSc.SNES().create()
        self.snes.setFunction(self.petsc_function.snes_mult, self.b)
#         self.snes.setJacobian(self.updateJacobian, self.Jmf, self.J)
        self.snes.setJacobian(self.updateJacobian, self.J)
        self.snes.setFromOptions()
        self.snes.getKSP().setType('gmres')
#        self.snes.getKSP().getPC().setType('bjacobi')
        self.snes.getKSP().getPC().setType('asm')
#         self.snes.getKSP().setType('preonly')
#         self.snes.getKSP().getPC().setType('lu')
#         self.snes.getKSP().getPC().setFactorSolverPackage(solver_package)
        
#         self.snes_nsp = PETSc.NullSpace().create(vectors=(self.x_nvec,))
#         self.snes.getKSP().setNullSpace(self.snes_nsp)
        
        
        # create Poisson object
        self.poisson_mat = PETScPoissonMatrix(self.da1, self.dax, 
                                              self.nx, self.nv, self.hx, self.hv,
                                              self.charge)
        
        # initialise Poisson matrix
        self.poisson_A = self.dax.createMat()
        self.poisson_A.setOption(self.poisson_A.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.poisson_A.setUp()
        
        # create linear Poisson solver
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_A)
        self.poisson_ksp.setType('preonly')
        self.poisson_ksp.getPC().setType('lu')
        self.poisson_ksp.getPC().setFactorSolverPackage(solver_package)
        
        self.poisson_nsp = PETSc.NullSpace().create(vectors=(self.p_nvec,))
        self.poisson_ksp.setNullSpace(self.poisson_nsp)
        
        
        
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
            print("Lx   = %e" % (L))
            print("vMin = %e" % (vMin))
            print("vMax = %e" % (vMax))
            print()
            print("nu   = %e" % (self.coll_freq))
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
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print()
        
        # normalise f to fit density and copy f to x
        nave = self.f.sum() * self.hv / self.nx
        self.f.scale(1./nave)
        self.copy_f_to_x()
        
        # calculate potential and moments
        self.calculate_moments()
        
        
        # check for external potential
        self.external = None
        if self.cfg['initial_data']['external_python'] != None:
            external_data = __import__("runs." + self.cfg['initial_data']['external_python'], globals(), locals(), ['external'], 0)
            self.external = external_data.external
        
        self.p_ext.set(0.)
        self.calculate_external(0.)
        
        
        # copy external potential
        self.petsc_jacobian_mf.update_external(self.p_ext)
        self.petsc_jacobian.update_external(self.p_ext)
        self.petsc_function.update_external(self.p_ext)
        self.petsc_matrix.update_external(self.p_ext)
        
        # update solution history
        self.petsc_jacobian_mf.update_history(self.f, self.h1)
        self.petsc_jacobian.update_history(self.f, self.h1)
        self.petsc_function.update_history(self.f, self.h1, self.p, self.n, self.nu, self.ne, self.u, self.e, self.a)
        self.petsc_matrix.update_history(self.f, self.h1, self.p, self.n, self.nu, self.ne, self.u, self.e, self.a)
        self.arakawa_gear.update_history(self.f, self.h1)
        
        
        # create HDF5 output file
        self.hdf5_viewer = PETSc.Viewer().createHDF5(hdf_out_filename,
                                          mode=PETSc.Viewer.Mode.WRITE,
                                          comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.HDF5PushGroup("/")
        
        # write grid data to hdf5 file
        coords_x.setName('x')
        coords_v.setName('v')

        self.hdf5_viewer(coords_x)
        self.hdf5_viewer(coords_v)
        
        # write initial data to hdf5 file
        self.hdf5_viewer(n0)
        self.hdf5_viewer(T0)
        
        # save to hdf5
        self.hdf5_viewer.HDF5SetTimestep(0)
        self.save_hdf5_vectors()
        
        
    def calculate_moments(self, potential=True):
        self.toolbox.compute_density(self.f, self.n)
        self.toolbox.compute_velocity_density(self.f, self.nu)
        self.toolbox.compute_energy_density(self.f, self.ne)
        
        self.calculate_velocity()             # calculate mean velocity
        self.calculate_energy()               # calculate mean energy
        self.calculate_collision_factor()     # calculate collision denominator
 
        self.copy_n_to_x()                    # copy density to solution vector
        self.copy_u_to_x()                    # copy velocity to solution vector
        self.copy_e_to_x()                    # copy energy to solution vector
        
        if potential:
            self.calculate_potential()            # calculate initial potential
            self.copy_p_to_x()                    # copy potential to solution vector
            self.copy_p_to_h()
    
    
    def calculate_potential(self, output=True):
        
        self.poisson_mat.formMat(self.poisson_A)
        self.poisson_mat.formRHS(self.n, self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        if output:
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Poisson:                            sum(phi) = %24.16E" % (phisum))
    
        
    def calculate_velocity(self):
        n_arr  = self.dax.getVecArray(self.n )[...]
        u_arr  = self.dax.getVecArray(self.u )[...]
        nu_arr = self.dax.getVecArray(self.nu)[...]
        
#        u_arr[:] = nu_arr / n_arr
        u_arr[:] = 0.
        
    
    def calculate_energy(self):
        n_arr  = self.dax.getVecArray(self.n )[...]
        e_arr  = self.dax.getVecArray(self.e )[...]
        ne_arr = self.dax.getVecArray(self.ne)[...]
        
#        e_arr[:] = ne_arr / n_arr
        e_arr[:] = 0.

    
    def calculate_collision_factor(self):
        u_arr = self.dax.getVecArray(self.u)[...]
        e_arr = self.dax.getVecArray(self.e)[...]
        a_arr = self.dax.getVecArray(self.a)[...]
        
#        a_arr[:] = 1. / ( e_arr - u_arr**2 )
        a_arr[:] = 0.
        
    
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
        x_arr = self.da2.getVecArray(self.x)[...]
        f_arr = self.da1.getVecArray(self.f)[...]
        
        f_arr[:, :] = x_arr[:, 0:self.nv] 
        
    
    def copy_f_to_x(self):
        x_arr = self.da2.getVecArray(self.x)[...]
        f_arr = self.da1.getVecArray(self.f)[...]
        
        x_arr[:, 0:self.nv] = f_arr[:, :] 
    
    
    def copy_x_to_p(self):
        x_arr = self.da2.getVecArray(self.x)[...]
        p_arr = self.dax.getVecArray(self.p)[...]
        
        p_arr[:] = x_arr[:, self.nv]
        
    
    def copy_p_to_x(self):
        p_arr = self.dax.getVecArray(self.p)[...]
        x_arr = self.da2.getVecArray(self.x)[...]
        
        x_arr[:, self.nv] = p_arr[:]
        
        
    def copy_p_to_h(self):
        p_arr = self.dax.getVecArray(self.p )[...]
        h_arr = self.da1.getVecArray(self.h1)[...]
        
        phisum = self.p.sum()
        phiave = phisum / self.nx
        
        for j in range(0, self.nv):
            h_arr[:, j] = p_arr[:] - phiave
        

    def copy_pext_to_h(self):
        p_arr = self.dax.getVecArray(self.p_ext)[...]
        h_arr = self.da1.getVecArray(self.h2   )[...]
    
        phisum = self.p_ext.sum()
        phiave = phisum / self.nx
        
        for j in range(0, self.nv):
            h_arr[:, j] = p_arr[:] - phiave
        

    def copy_x_to_n(self):
        x_arr  = self.da2.getVecArray(self.x )[...]
        n_arr  = self.dax.getVecArray(self.n )[...]
        nu_arr = self.dax.getVecArray(self.nu)[...]
        ne_arr = self.dax.getVecArray(self.ne)[...]
        
        n_arr [:] = x_arr[:, self.nv+1]
        nu_arr[:] = x_arr[:, self.nv+2]
        ne_arr[:] = x_arr[:, self.nv+3]
        
    
    def copy_n_to_x(self):
        n_arr  = self.dax.getVecArray(self.n )[...]
        nu_arr = self.dax.getVecArray(self.nu)[...]
        ne_arr = self.dax.getVecArray(self.ne)[...]
        x_arr  = self.da2.getVecArray(self.x )[...]
        
        x_arr[:, self.nv+1] = n_arr[:]
        x_arr[:, self.nv+2] = nu_arr[:]
        x_arr[:, self.nv+3] = ne_arr[:]
        
        
    def copy_x_to_u(self):
        x_arr = self.da2.getVecArray(self.x)[...]
        u_arr = self.dax.getVecArray(self.u)[...]
        
        u_arr[:] = x_arr[:, self.nv+4]
        
    
    def copy_u_to_x(self):
        u_arr = self.dax.getVecArray(self.u)[...]
        x_arr = self.da2.getVecArray(self.x)[...]
        
        x_arr[:, self.nv+4] = u_arr[:]
        
        
    def copy_x_to_e(self):
        x_arr = self.da2.getVecArray(self.x)[...]
        e_arr = self.dax.getVecArray(self.e)[...]
        
        e_arr[:] = x_arr[:, self.nv+5]
        
    
    def copy_e_to_x(self):
        e_arr = self.dax.getVecArray(self.e)[...]
        x_arr = self.da2.getVecArray(self.x)[...]
        
        x_arr[:, self.nv+5] = e_arr[:]
        
        
    def save_to_hdf5(self, itime):
        # save to hdf5 file
        if itime % self.nsave == 0 or itime == self.nt + 1:
            self.hdf5_viewer.HDF5SetTimestep(self.hdf5_viewer.HDF5GetTimestep() + 1)
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

    
    def updateMatrix(self, snes, X, J, P):
        self.petsc_matrix.formMat(J)
        J.setNullSpace(self.nullspace)
        
        if J != P:
            self.petsc_matrix.formMat(P)
            P.setNullSpace(self.nullspace)
    
    
    def updateJacobian(self, snes, X, J, P):
        self.petsc_jacobian_mf.update_previous(X)
        self.petsc_jacobian.update_previous(X)
        
        self.petsc_jacobian.formMat(J)
        J.setNullSpace(self.nullspace)
        
        if J != P:
            self.petsc_jacobian.formMat(P)
            P.setNullSpace(self.nullspace)
        
    
    def initial_guess_rk4(self):
        # calculate initial guess for distribution function

        for i in range(0, self.nInitial):
#             self.arakawa_rk4.rk4_J1(self.f, self.h1)
#             self.arakawa_rk4.rk4_J2(self.f, self.h1)
            self.arakawa_rk4.rk4_J4(self.f, self.h1)
            
            self.copy_f_to_x()
            
            self.calculate_moments(potential=False)
            self.calculate_potential(output=False)
            
            self.copy_p_to_x()
            self.copy_p_to_h()
        
        
        self.petsc_function.mult(self.x, self.b)
        ignorm = self.b.norm()
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  RK4 Initial Guess:                  funcnorm = %24.16E" % (ignorm))
            print("                                      sum(phi) = %24.16E" % (phisum))
         
    
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
                
                self.calculate_moments(potential=False)
                self.calculate_potential(output=False)
                
                self.copy_p_to_x()
                self.copy_p_to_h()
                
                if i < self.nInitial - 1:
                    self.arakawa_gear.update_history(self.f, self.h1)
            
            
            self.petsc_function.mult(self.x, self.b)
            ignorm = self.b.norm()
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Gear Initial Guess:                 funcnorm = %24.16E" % (ignorm))
                print("                                      sum(phi) = %24.16E" % (phisum))
         
        
        
    
    def initial_guess(self):
        self.snes_linear.solve(None, self.x)
        self.copy_x_to_data()
        
        self.calculate_moments(potential=False)
        
        self.petsc_function.mult(self.x, self.b)
        ignorm = self.b.norm()
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("  Linear Solver:                      funcnorm = %24.16E" % (ignorm))
            print("                                      sum(phi) = %24.16E" % (phisum))
        
    
    def run(self):
        for itime in range(1, self.nt+1):
            current_time = self.ht*itime
            
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, current_time, localtime) )
                print
                self.time.setValue(0, current_time)
            
            # calculate external field and copy to matrix, jacobian and function
            self.calculate_external(current_time)
            self.petsc_jacobian_mf.update_external(self.p_ext)
            self.petsc_jacobian.update_external(self.p_ext)
            self.petsc_function.update_external(self.p_ext)
            self.petsc_matrix.update_external(self.p_ext)
            
            
            self.x.copy(self.xh)
            
            self.petsc_function.mult(self.x, self.b)
            prev_norm = self.b.norm()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Previous Step:                      funcnorm = %24.16E" % (prev_norm))
            
            # calculate initial guess via RK4
#             self.initial_guess_rk4()
            
            # calculate initial guess via Gear
            self.initial_guess_gear(itime)
            
            # check if residual went down
#             self.petsc_function.mult(self.x, self.b)
#             ig_norm = self.b.norm()
#             
#             if ig_norm > prev_norm:
#                 self.xh.copy(self.x)
            
            
            # calculate initial guess via linear solver
#            self.initial_guess()
            
            
            # nonlinear solve
            self.snes.solve(None, self.x)
            
            # output some solver info
            if PETSc.COMM_WORLD.getRank() == 0:
                print("  Nonlin Solver:  %5i iterations,   funcnorm = %24.16E" % (self.snes.getIterationNumber(), self.snes.getFunctionNorm()) )
                print()
            
            if self.snes.getConvergedReason() < 0:
                if PETSc.COMM_WORLD.getRank() == 0:
                    print()
                    print("Solver not converging...   %i" % (self.snes.getConvergedReason()))
                    print()
            
            
#             if PETSc.COMM_WORLD.getRank() == 0:
#                 mat_viewer = PETSc.Viewer().createDraw(size=(800,800), comm=PETSc.COMM_WORLD)
#                 mat_viewer(self.J)
#                 
#                 print
#                 input('Hit any key to continue.')
#                 print
            
            
            # update data vectors
            self.copy_x_to_data()
            self.calculate_velocity()
            self.calculate_energy()
            self.calculate_collision_factor()
            
            # update history
            self.petsc_jacobian_mf.update_history(self.f, self.h1)
            self.petsc_jacobian.update_history(self.f, self.h1)
            self.petsc_function.update_history(self.f, self.h1, self.p, self.n, self.nu, self.ne, self.u, self.e, self.a)
            self.petsc_matrix.update_history(self.f, self.h1, self.p, self.n, self.nu, self.ne, self.u, self.e, self.a)
            self.arakawa_gear.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
            
#            # some solver output
            phisum = self.p.sum()
#            
#            
            if PETSc.COMM_WORLD.getRank() == 0:
#                print("     Solver")
                print("     sum(phi) = %24.16E" % (phisum))
##                print("     Solver:   %5i iterations,   sum(phi) = %24.16E" % (phisum))
##                print("                               res(solver)  = %24.16E" % (res_solver))
##                print("                               res(vlasov)  = %24.16E" % (res_vlasov))
##                print("                               res(poisson) = %24.16E" % (res_poisson))
##                print
                
            
            

    def check_jacobian(self):
        
        (xs, xe), = self.da1.getRanges()
        
        eps = 1.E-7
        
        # update previous iteration
        self.petsc_jacobian.update_previous(self.x)
        
        # calculate jacobian
        self.petsc_jacobian.formMat(self.J)
        
        J = self.J
        
        
        # create working vectors
        Jx  = self.da2.createGlobalVec()
        dJ  = self.da2.createGlobalVec()
        ex  = self.da2.createGlobalVec()
        dx  = self.da2.createGlobalVec()
        dF  = self.da2.createGlobalVec()
        Fxm = self.da2.createGlobalVec()
        Fxp = self.da2.createGlobalVec()
        
        
#         sx = -2
#         sx = -1
        sx =  0
#         sx = +1
#         sx = +2
        
        nfield=self.nv+4
        
        for ifield in range(0, nfield):
            for ix in range(xs, xe):
                for tfield in range(0, nfield):
                    
                    # compute ex
                    ex_arr = self.da2.getVecArray(ex)
                    ex_arr[:] = 0.
                    ex_arr[(ix+sx) % self.nx, ifield] = 1.
                    
                    
                    # compute J.e
                    J.mult(ex, dJ)
                    
                    dJ_arr = self.da2.getVecArray(dJ)
                    Jx_arr = self.da2.getVecArray(Jx)
                    Jx_arr[ix, tfield] = dJ_arr[ix, tfield]
                    
                    
                    # compute F(x - eps ex)
                    self.x.copy(dx)
                    dx_arr = self.da2.getVecArray(dx)
                    dx_arr[(ix+sx) % self.nx, ifield] -= eps
                    
                    self.petsc_function.mult(dx, Fxm)
                    
                    
                    # compute F(x + eps ex)
                    self.x.copy(dx)
                    dx_arr = self.da2.getVecArray(dx)
                    dx_arr[(ix+sx) % self.nx, ifield] += eps
                    
                    self.petsc_function.mult(dx, Fxp)
                    
                    
                    # compute dF = [F(x + eps ex) - F(x - eps ex)] / (2 eps)
                    Fxm_arr = self.da2.getVecArray(Fxm)
                    Fxp_arr = self.da2.getVecArray(Fxp)
                    dF_arr  = self.da2.getVecArray(dF)
                    
                    dF_arr[ix, tfield] = ( Fxp_arr[ix, tfield] - Fxm_arr[ix, tfield] ) / (2. * eps)
                        
            
            diff = np.zeros(nfield)
            
            for tfield in range(0,nfield):
#                print()
#                print("Fields: (%5i, %5i)" % (ifield, tfield))
#                print()
                
                Jx_arr = self.da2.getVecArray(Jx)[...][:, tfield]
                dF_arr = self.da2.getVecArray(dF)[...][:, tfield]
                
                
#                 print("Jacobian:")
#                 print(Jx_arr)
#                 print()
#                   
#                 print("[F(x+dx) - F(x-dx)] / [2 eps]:")
#                 print(dF_arr)
#                 print()
#                 
#                 print("Difference:")
#                 print(Jx_arr - dF_arr)
#                 print()
                
                
#                if ifield == 3 and tfield == 2:
#                    print("Jacobian:")
#                    print(Jx_arr)
#                    print()
#                    
#                    print("[F(x+dx) - F(x-dx)] / [2 eps]:")
#                    print(dF_arr)
#                    print()
                
                
                diff[tfield] = (Jx_arr - dF_arr).max()
            
            print()
        
            for tfield in range(0,nfield):
                print("max(difference)[field=%2i, equation=%2i] = %16.8E" % ( ifield, tfield, diff[tfield] ))
            
            print()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1D(args.runfile)
    petscvp.run()
#     petscvp.check_jacobian()
    
