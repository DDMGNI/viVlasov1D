'''
Created on Mar 23, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

import argparse
import time

from core import Config
from data import maxwellian

from vlasov.predictor.PETScArakawaRK4     import PETScArakawaRK4
from vlasov.predictor.PETScPoisson        import PETScPoissonSolver
from vlasov.predictor.PETScVlasovFunction import PETScVlasovFunction
from vlasov.predictor.PETScVlasovJacobian import PETScVlasovJacobian



class petscVP1D(object):
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
        nx   = cfg['grid']['nx']                    # number of points in x
        nv   = cfg['grid']['nv']                    # number of points in v
        L    = cfg['grid']['L']
        vMin = cfg['grid']['vmin']
        vMax = cfg['grid']['vmax']
        
        self.nx = nx
        self.nv = nv
        
        self.hx = L / nx                            # gridstep size in x
        self.hv = (vMax - vMin) / (nv-1)            # gridstep size in v
        
        self.time = PETSc.Vec().createMPI(1, 1, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        self.tolerance = cfg['solver']['petsc_residual']
#        self.tolerance = 1E-5
#        self.tolerance = 1E-7

        self.max_iter  = 10
#        self.max_iter  = 100
#        self.max_iter  = 1000
        
        OptDB.setValue('ksp_rtol', self.tolerance)
        OptDB.setValue('ksp_max_it', self.max_iter)


#        OptDB.setValue('ksp_monitor', '')
#        OptDB.setValue('log_info', '')
#        OptDB.setValue('log_summary', '')

#        OptDB.setValue('ksp_gmres_classicalgramschmidt', '')
#        OptDB.setValue('ksp_gmres_modifiedgramschmidt', '')
#        OptDB.setValue('ksp_gmres_restart', 3)
        
        
        # create DA with single dof
        self.da1 = PETSc.DA().create(dim=2, dof=1,
                                    sizes=[nx, nv],
                                    proc_sizes=[PETSc.COMM_WORLD.getSize(), 1],
                                    boundary_type=('periodic', 'none'),
                                    stencil_width=1,
                                    stencil_type='box')
        
        
        # create DA for Poisson guess
        self.dax = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[nx],
                                    proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                    boundary_type=('periodic'),
                                    stencil_width=1,
                                    stencil_type='box')
        
        # create DA for y grid
        self.day = PETSc.DA().create(dim=1, dof=1,
                                    sizes=[nv],
                                    proc_sizes=[PETSc.COMM_WORLD.getSize()],
                                    boundary_type=('none'))
        
        
        # initialise grid
        self.da1.setUniformCoordinates(xmin=0.0,  xmax=L,
                                       ymin=vMin, ymax=vMax)
        
        self.dax.setUniformCoordinates(xmin=0.0, xmax=L)
        
        self.day.setUniformCoordinates(xmin=vMin, xmax=vMax) 
        
        
        # create residual, solution and RHS vector for Vlasov solver
        self.df = self.da1.createGlobalVec()
        self.f  = self.da1.createGlobalVec()
        self.fb = self.da1.createGlobalVec()
        
        # create solution and RHS vector for Poisson solver
        self.p  = self.dax.createGlobalVec()
        self.pb = self.dax.createGlobalVec()
        
        # create vectors for Hamiltonians, distribution functions,
        # density and the potential
        self.h0 = self.da1.createGlobalVec()
        self.h1 = self.da1.createGlobalVec()
        self.f  = self.da1.createGlobalVec()
        self.n  = self.dax.createGlobalVec()
        
        # set variable names
        self.h0.setName('h0')
        self.h1.setName('h1')
        self.f.setName('f')
        self.n.setName('n')
        self.p.setName('phi')
        
        
        # create Matrix object
        self.vlasov_function = PETScVlasovFunction(self.da1, self.dax, self.h0, 
                                                   nx, nv, self.ht, self.hx, self.hv)
        
        self.vlasov_jacobian = PETScVlasovJacobian(self.da1, self.dax, self.h0, 
                                                   nx, nv, self.ht, self.hx, self.hv)
        
        self.poisson_matrix = PETScPoissonSolver(self.da1, self.dax, 
                                                 nx, nv, self.hx, self.hv,
                                                 cfg['solver']['poisson_const'])
        
        self.J = PETSc.Mat().createPython([self.f.getSizes(), self.fb.getSizes()], comm=PETSc.COMM_WORLD)
        self.J.setPythonContext(self.vlasov_jacobian)
        self.J.setUp()
        
        self.P = PETSc.Mat().createPython([self.p.getSizes(), self.pb.getSizes()], comm=PETSc.COMM_WORLD)
        self.P.setPythonContext(self.poisson_matrix)
        self.P.setUp()
        
        
        # create linear solver and preconditioner
        self.vlasov_ksp = PETSc.KSP().create()
        self.vlasov_ksp.setOperators(self.J)
        self.vlasov_ksp.setFromOptions()
        self.vlasov_ksp.setType('gmres')
        self.vlasov_ksp.getPC().setType('none')
#        self.vlasov_ksp.setInitialGuessNonzero(True)
        
        
        # create Poisson matrix and solver
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setOperators(self.P)
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setType('cg')
        self.poisson_ksp.getPC().setType('none')
#        self.poisson_ksp.setInitialGuessNonzero(True)
        
        
        # create Arakawa RK4 solver object
        self.arakawa_rk4 = PETScArakawaRK4(self.da1, self.h0, nx, nv, self.ht, self.hx, self.hv)
        
        
        # print some info
        if PETSc.COMM_WORLD.getRank() == 0:
            print
            print("ht = %e" % (self.ht))
            print("hx = %e" % (self.hx))
            print("hv = %e" % (self.hv))
            print
            print("CFL = %e" % (self.hx / vMax))
            print
        
        
        # set initial data
        n0 = self.dax.createGlobalVec()
        T0 = self.dax.createGlobalVec()
        
        n0_arr = self.dax.getVecArray(n0)
        T0_arr = self.dax.getVecArray(T0)
        
        n0.setName('n0')
        T0.setName('T0')
        
        f_arr = self.da1.getVecArray(self.f)
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        coords = self.da1.getCoordinateDA().getVecArray(self.da1.getCoordinates())
        
        
        if cfg['initial_data']['distribution_python'] != None:
            init_data = __import__("runs." + cfg['initial_data']['distribution_python'], globals(), locals(), ['distribution'], 0)
            
            for i in range(xs, xe):
                for j in range(ys, ye):
                    if j == 0 or j == nv-1:
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
                    if j == 0 or j == nv-1:
                        f_arr[i,j] = 0.0
                    else:
                        f_arr[i,j] = n0_arr[i] * maxwellian(T0_arr[i], coords[i,j][1])
        
        # normalise f to fit density
        ### TODO (?) ###
        
        # calculate density
        n_arr  = self.dax.getVecArray(self.n)
        n_arr[xs:xe] = f_arr[xs:xe].sum(axis=1) * self.hv
        
        
        # initialise kinetic hamiltonian
        h0_arr = self.da1.getVecArray(self.h0)
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                h0_arr[i, j] = 0.5 * coords[i,j][1]**2 # * self.mass
        
        
        # solve initial potential
        self.calculate_potential()
        
        
        # update solution history
        self.vlasov_jacobian.update_history(self.f, self.h1)
        self.vlasov_function.update_history(self.f, self.h1)
        
        
        # create HDF5 output file
        self.hdf5_viewer = PETSc.Viewer().createHDF5(cfg['io']['hdf5_output'],
                                          mode=PETSc.Viewer.Mode.WRITE,
                                          comm=PETSc.COMM_WORLD)
        
        self.hdf5_viewer.HDF5PushGroup("/")
        
        
        # write grid data to hdf5 file
        coords_x = self.dax.getCoordinates()
        coords_y = self.day.getCoordinates()
        
        coords_x.setName('x')
        coords_y.setName('v')

        self.hdf5_viewer(coords_x)
        self.hdf5_viewer(coords_y)
        
        
        # write initial data to hdf5 file
        self.hdf5_viewer(n0)
        self.hdf5_viewer(T0)
        
        self.hdf5_viewer.HDF5SetTimestep(0)
        self.hdf5_viewer(self.time)
        self.hdf5_viewer(self.f)
        self.hdf5_viewer(self.n)
        self.hdf5_viewer(self.p)
        self.hdf5_viewer(self.h0)
        self.hdf5_viewer(self.h1)
        
        
    
    def __del__(self):
        del self.hdf5_viewer
        
    
    
    def run(self):
        for itime in range(1, self.nt+1):
            if PETSc.COMM_WORLD.getRank() == 0:
                localtime = time.asctime( time.localtime(time.time()) )
                print("\ni = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            
            # calculate initial guess for distribution function
            self.initial_guess()
            
#            for iter in range(0, self.max_iter):
#                # check if norm is smaller than tolerance
#                if norm < self.tolerance:
#                    break
            
            
            print
            print("     iter = %3i" % (1)) 
#            print("     iter = %3i" % (iter)) 
            
            # update previous iteration
            self.vlasov_jacobian.update_previous(self.f, self.h1)
        
            # calculate function and norm
            self.vlasov_function.matrix_mult(self.f, self.h1, self.fb)
            norm0 = self.fb.norm()
            
            # RHS = - function
            self.fb.scale(-1.)
            
            # solve
            self.df.set(0.)
            self.vlasov_ksp.solve(self.fb, self.df)
            
            # add to solution vector
            self.f.axpy(1., self.df)
            
            # calculate function and norm
            self.vlasov_function.matrix_mult(self.f, self.h1, self.fb)
            norm1 = self.fb.norm()
            
            # update potential
            self.calculate_potential()
            
            # some solver output
            if PETSc.COMM_WORLD.getRank() == 0:
                print("     Vlasov Solver:   %5i iterations,   residual = %24.16E " % (self.vlasov_ksp.getIterationNumber(), self.vlasov_ksp.getResidualNorm()) )
                print("                                         Initial Function Norm = %24.16E" % (norm0) )
                print("                                         Final   Function Norm = %24.16E" % (norm1) )
                
            
            # update history
            self.vlasov_function.update_history(self.f, self.h1)
            self.vlasov_jacobian.update_history(self.f, self.h1)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
            # some solver output
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("   Nonlin Solver:       %5i iterations,   tolerance = %E " % (1, self.tolerance) )
#                print("   Nonlin Solver:  %5i iterations,   tolerance = %E " % (iter, self.tolerance) )
                print("        sum(phi):       %24.16E" % (phisum))
                print
            
        
    
    def initial_guess(self):
        # calculate initial guess for distribution function
        self.arakawa_rk4.rk4(self.f, self.h1)
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     RK4 predictor")
        
        self.calculate_potential()
        
        
        
    def calculate_potential(self):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # calculate initial guess for potential
        self.poisson_matrix.formRHS(self.f, self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        # copy potential to hamiltonian
        p_arr  = self.dax.getVecArray(self.p)
        h1_arr = self.da1.getVecArray(self.h1)

        phisum = self.p.sum()
        phiave = phisum / self.nx

        p_arr[xs:xe] -= phiave
        
        for j in range(ys, ye):
            h1_arr[xs:xe, j] = p_arr[xs:xe]
        
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("     Poisson:         %5i iterations,   residual = %24.16E" % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
            print("                                          sum(phi) = %24.16E" % (phisum))
        
    
    
    def save_to_hdf5(self, itime):
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        # copy solution to f and p vectors
        f_arr  = self.da1.getVecArray(self.f)
        n_arr  = self.dax.getVecArray(self.n)

        n_arr[xs:xe] = f_arr[xs:xe, :].sum(axis=1) * self.hv
        
        
        # save to hdf5 file
        if itime % self.nsave == 0 or itime == self.nt + 1:
            self.hdf5_viewer.HDF5SetTimestep(self.hdf5_viewer.HDF5GetTimestep() + 1)
            self.hdf5_viewer(self.time)
            self.hdf5_viewer(self.f)
            self.hdf5_viewer(self.n)
            self.hdf5_viewer(self.p)
            self.hdf5_viewer(self.h0)
            self.hdf5_viewer(self.h1)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PETSc Vlasov-Poisson Solver in 1D')
    parser.add_argument('runfile', metavar='runconfig', type=str,
                        help='Run Configuration File')
    
    args = parser.parse_args()
    
    petscvp = petscVP1D(args.runfile)
    petscvp.run()
    
