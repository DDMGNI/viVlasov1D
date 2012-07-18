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

from vlasov.predictor.PETScArakawaRK4 import PETScArakawaRK4
from vlasov.predictor.PETScPoisson    import PETScPoissonSolver
from vlasov.predictor.PETScVlasov     import PETScVlasovSolver


#from vlasov.vi.sbs_sym_arakawa_1st.petsc_sparse_simple     import PETScSolver
#from vlasov.vi.sbs_sym_arakawa_1st.PETScMatrixFree         import PETScSolver
from vlasov.vi.sbs_sym_arakawa_1st.PETScMatrixFreeSimple          import PETScSolver
#from vlasov.vi.sbs_sym_arakawa_1st.PETScMatrixFreeNonlinearSimple import PETScSolver




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
        
        self.hx = L / nx                            # gridstep size in x
        self.hv = (vMax - vMin) / (nv-1)            # gridstep size in v
        
        self.time = PETSc.Vec().createMPI(1, 1, comm=PETSc.COMM_WORLD)
        self.time.setName('t')
        
        if PETSc.COMM_WORLD.getRank() == 0:
            self.time.setValue(0, 0.0)
        
        
        # set some PETSc options
        OptDB = PETSc.Options()
        
        OptDB.setValue('ksp_rtol', cfg['solver']['petsc_residual'])
        OptDB.setValue('ksp_max_it', 100)
#        OptDB.setValue('ksp_max_it', 200)
#        OptDB.setValue('ksp_max_it', 1000)

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
        
        
        # create DA (dof = number of species + 1 for the potential)
        self.da2 = PETSc.DA().create(dim=2, dof=2,
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
        
        self.da2.setUniformCoordinates(xmin=0.0,  xmax=L,
                                       ymin=vMin, ymax=vMax)
        
        self.dax.setUniformCoordinates(xmin=0.0, xmax=L)
        
        self.day.setUniformCoordinates(xmin=vMin, xmax=vMax) 
        
        
        # create solution and RHS vector
        self.x  = self.da2.createGlobalVec()
        self.b  = self.da2.createGlobalVec()
        
        # create solution and RHS vector for Vlasov and Poisson solver
        self.vb = self.da1.createGlobalVec()
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
        
        
        # create Matrix object
#        _solver = __import__(cfg['solver']['solver_module'], fromlist=['PETScSolver'])
#        self.vp = _solver.Solver(self.grid, cfg['solver']['solver_method'])
        
#        self.petsc_mat = _solver.PETScSolver(self.da, self.x, self.b,
        self.petsc_mat = PETScSolver(self.da1, self.da2, self.dax,
                                     self.x, self.b, self.h0, 
                                     nx, nv, self.ht, self.hx, self.hv,
                                     cfg['solver']['poisson_const'])
        
        # create sparse matrix
        if self.petsc_mat.isSparse():
            self.A = self.da2.getMatrix('aij')
        
        else:
            self.A = PETSc.Mat().createPython([self.x.getSizes(), self.b.getSizes()], comm=PETSc.COMM_WORLD)
            self.A.setPythonContext(self.petsc_mat)
            self.A.setUp()

        # create linear solver and preconditioner
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.A)
        self.ksp.setType(cfg['solver']['petsc_ksp_type'])
        self.ksp.setInitialGuessNonzero(True)
        
        self.pc = self.ksp.getPC()
        self.pc.setType(cfg['solver']['petsc_pc_type'])
        
        
        # create Vlasov matrix and solver
        self.vlasov_mat = PETScVlasovSolver(self.da1, self.da2, self.h0, 
                                            nx, nv, self.ht, self.hx, self.hv)
        
        self.vlasov_A = PETSc.Mat().createPython([self.f.getSizes(), self.vb.getSizes()], comm=PETSc.COMM_WORLD)
        self.vlasov_A.setPythonContext(self.vlasov_mat)
        self.vlasov_A.setUp()
        
        self.vlasov_ksp = PETSc.KSP().create()
        self.vlasov_ksp.setFromOptions()
        self.vlasov_ksp.setOperators(self.vlasov_A)
        self.vlasov_ksp.setType(cfg['solver']['petsc_ksp_type'])
        self.vlasov_ksp.setInitialGuessNonzero(True)
        
        self.vlasov_pc = self.vlasov_ksp.getPC()
        self.vlasov_pc.setType('none')
        
        
        # create Poisson matrix and solver
        self.poisson_mat = PETScPoissonSolver(self.da1, self.dax, self.f, 
                                              nx, nv, self.hx, self.hv,
                                              cfg['solver']['poisson_const'])
        
        self.poisson_A = PETSc.Mat().createPython([self.p.getSizes(), self.pb.getSizes()], comm=PETSc.COMM_WORLD)
        self.poisson_A.setPythonContext(self.poisson_mat)
        self.poisson_A.setUp()
        
        self.poisson_ksp = PETSc.KSP().create()
        self.poisson_ksp.setFromOptions()
        self.poisson_ksp.setOperators(self.poisson_A)
        self.poisson_ksp.setType(cfg['solver']['petsc_ksp_type'])
        self.poisson_ksp.setInitialGuessNonzero(True)
        
        self.poisson_pc = self.poisson_ksp.getPC()
        self.poisson_pc.setType('none')
        
        
        # create Arakawa RK4 solver object
        self.arakawa_rk4 = PETScArakawaRK4(self.da1, self.da2, self.h0, nx, nv, self.ht, self.hx, self.hv)
        
        
        # set initial data
        n0 = self.dax.createGlobalVec()
        T0 = self.dax.createGlobalVec()
        
        n0_arr = self.dax.getVecArray(n0)
        T0_arr = self.dax.getVecArray(T0)
        
        n0.setName('n0')
        T0.setName('T0')
        
        f_arr = self.da1.getVecArray(self.f)
        x_arr = self.da2.getVecArray(self.x)
        
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        coords  = self.da2.getCoordinateDA().getVecArray(self.da2.getCoordinates())
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print
            print("ht = %e" % (self.ht))
            print("hx = %e" % (self.hx))
            print("hv = %e" % (self.hv))
            print
            print("CFL = %e" % (self.hx / vMax))
            print
        
#        print(coords[1,0][0] - coords[0,0][0])
#        print(coords[0,1][1] - coords[0,0][1])
#        
#        print(L)
#        print(L-self.hx)
#        print(coords[...][-1,0][0])
        
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
        
        # copy distribution function to solution vector
        x_arr[xs:xe, ys:ye, 0] = f_arr[xs:xe, ys:ye]
        
        # calculate density
        n_arr  = self.dax.getVecArray(self.n)
        n_arr[xs:xe] = f_arr[xs:xe].sum(axis=1) * self.hv
        
        
        # initialise kinetic hamiltonian
        h0_arr = self.da1.getVecArray(self.h0)
        
        for i in range(xs, xe):
            for j in range(ys, ye):
                h0_arr[i, j] = 0.5 * coords[i,j][1]**2 # * self.mass
        
        
        # solve initial potential
        self.poisson_mat.formRHS(self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        p_arr  = self.dax.getVecArray(self.p)
        x_arr  = self.da2.getVecArray(self.x)
        h1_arr = self.da1.getVecArray(self.h1)
        
        for j in range(ys, ye):
            h1_arr[xs:xe, j]    = p_arr[xs:xe]
            x_arr [xs:xe, j, 1] = p_arr[xs:xe]
        
        
        # update solution history
        self.petsc_mat.update_history(self.x)
        self.vlasov_mat.update_history(self.x)
        
        
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
        self.hdf5_viewer(self.x)
        self.hdf5_viewer(self.b)
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
                print("\nit = %4d,   t = %10.4f,   %s" % (itime, self.ht*itime, localtime) )
                self.time.setValue(0, self.ht*itime)
            
            # build matrix
            if self.petsc_mat.isSparse():
                self.petsc_mat.formMat(self.A, self.x)
            
            # build RHS
            self.petsc_mat.formRHS(self.b)
            
            # calculate initial guess for distribution function
            self.initial_guess()
            
            # solve
            self.ksp.solve(self.b, self.x)
            
            # update history
            self.petsc_mat.update_history(self.x)
            self.vlasov_mat.update_history(self.x)
            
            # save to hdf5
            self.save_to_hdf5(itime)
            
            
            # some solver output
            phisum = self.p.sum()
            
            if PETSc.COMM_WORLD.getRank() == 0:
                print("   Vlasov:  %5i iterations,   residual = %24.16E " % (self.ksp.getIterationNumber(), self.ksp.getResidualNorm()) )
                print("                                sum(phi) = %24.16E" % (phisum))
                print
            
        
    
    def initial_guess(self):
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        # calculate initial guess for distribution function
        self.arakawa_rk4.rk4(self.x)
        
        x_arr  = self.da2.getVecArray(self.x)
        f_arr  = self.da1.getVecArray(self.f)
        
        f_arr[xs:xe, ys:ye]  = x_arr[xs:xe, ys:ye, 0] 
        
        
        # calculate initial guess for potential
        self.poisson_mat.formRHS(self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        p_arr = self.dax.getVecArray(self.p)
        x_arr = self.da2.getVecArray(self.x)
        
        for j in range(ys, ye):
            x_arr[xs:xe, j, 1] = p_arr[xs:xe]
        
        self.vlasov_mat.update_current(self.x)
        
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("   Poisson: %5i iterations,   residual = %24.16E " % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
            print("                                sum(phi) = %24.16E" % (phisum))
        
        
        # correct initial guess for distribution function
        self.vlasov_mat.formRHS(self.vb)
        self.vlasov_ksp.solve(self.vb, self.f)
        
        x_arr  = self.da2.getVecArray(self.x)
        f_arr  = self.da1.getVecArray(self.f)
        
        x_arr[xs:xe, ys:ye, 0] = f_arr[xs:xe, ys:ye] 
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("   Vlasov:  %5i iterations,   residual = %24.16E " % (self.vlasov_ksp.getIterationNumber(), self.vlasov_ksp.getResidualNorm()) )
            

        # correct initial guess for potential
        self.poisson_mat.formRHS(self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        p_arr = self.dax.getVecArray(self.p)
        x_arr = self.da2.getVecArray(self.x)
        
        for j in range(ys, ye):
            x_arr[xs:xe, j, 1] = p_arr[xs:xe]
        
        self.vlasov_mat.update_current(self.x)
        
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("   Poisson: %5i iterations,   residual = %24.16E " % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
            print("                                sum(phi) = %24.16E" % (phisum))
        
        
        # correct initial guess for distribution function
        self.vlasov_mat.update_current(self.x)
        
        self.vlasov_mat.formRHS(self.vb)
        self.vlasov_ksp.solve(self.vb, self.f)
        
        x_arr  = self.da2.getVecArray(self.x)
        f_arr  = self.da1.getVecArray(self.f)
        
        x_arr[xs:xe, ys:ye, 0] = f_arr[xs:xe, ys:ye] 
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("   Vlasov:  %5i iterations,   residual = %24.16E " % (self.vlasov_ksp.getIterationNumber(), self.vlasov_ksp.getResidualNorm()) )
            

        # correct initial guess for potential
        self.poisson_mat.formRHS(self.pb)
        self.poisson_ksp.solve(self.pb, self.p)
        
        p_arr = self.dax.getVecArray(self.p)
        x_arr = self.da2.getVecArray(self.x)
        
        for j in range(ys, ye):
            x_arr[xs:xe, j, 1] = p_arr[xs:xe]
        
        phisum = self.p.sum()
        
        if PETSc.COMM_WORLD.getRank() == 0:
            print("   Poisson: %5i iterations,   residual = %24.16E " % (self.poisson_ksp.getIterationNumber(), self.poisson_ksp.getResidualNorm()) )
            print("                                sum(phi) = %24.16E" % (phisum))            
    
    
    
    def save_to_hdf5(self, itime):
        (xs, xe), (ys, ye) = self.da2.getRanges()
        
        # copy solution to f and p vectors
        x_arr  = self.da2.getVecArray(self.x)
        f_arr  = self.da1.getVecArray(self.f)
        n_arr  = self.dax.getVecArray(self.n)
        p_arr  = self.dax.getVecArray(self.p)
        h1_arr = self.da1.getVecArray(self.h1)

        f_arr[xs:xe, ys:ye]  = x_arr[xs:xe, ys:ye, 0] 
        n_arr[xs:xe]         = x_arr[xs:xe, :, 0].sum(axis=1) * self.hv
        p_arr[xs:xe]         = x_arr[xs:xe, 0, 1]
        h1_arr[xs:xe, ys:ye] = x_arr[xs:xe, ys:ye, 1]
        
        
        # save to hdf5 file
        if itime % self.nsave == 0 or itime == self.grid.nt + 1:
            self.hdf5_viewer.HDF5SetTimestep(self.hdf5_viewer.HDF5GetTimestep() + 1)
            self.hdf5_viewer(self.time)
            self.hdf5_viewer(self.x)
            self.hdf5_viewer(self.b)
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
    
