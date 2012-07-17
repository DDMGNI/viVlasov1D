'''
Created on Apr 10, 2012

@author: mkraus
'''

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc


import numpy as np

from scipy.sparse import csc_matrix, eye, hstack, lil_matrix, spdiags, vstack
from scipy.sparse.linalg import spsolve


from poisson import Poisson
from vlasov  import Arakawa, VlasovSolver
from vlasov.sparse import create_periodic_band_matrix, create_tridiagonal_block_matrix


class Solver(VlasovSolver):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    
    def __init__(self, grid, ht, method=None, residual=1.0E-7, max_iter=100):
        '''
        Constructor
        '''
        
        # initialise parent class
        VlasovSolver.__init__(self, grid, ht)
        
        
        # set number of backward timesteps needed by this solver
        self.nhist = 2
        
        
        # set some PETSc options (necessary?)
        OptDB = PETSc.Options()
        
        OptDB.setInt('nx', grid.nx)
        OptDB.setInt('ny', grid.nv)
        
#        PETSc.Options().setValue('pc_factor_shift_type', 'POSITIVE_DEFINITE')
        
        
        # create DA
        self.da = PETSc.DA().create([grid.nx, grid.nv+1])
        
        assert self.da.getDim() == 2
        
        # create 2d data arrays
#        f = da.create2D(DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX,
#                        self.grid.nx, self.grid.nv, PETSC_DECIDE, 1, self.nhist+1)
#        h = da.create2D(DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX,
#                        self.grid.nx, self.grid.nv, PETSC_DECIDE, 1, self.nhist+1)
#        p = da.create1D(DMDA_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX,
#                        self.grid.nx, PETSC_DECIDE)
        
        # create solution and RHS vector
        x = da.createGlobalVec()
        b = da.createGlobalVec()
        
        # create sparse matrix
        self.A = PETSc.Mat().createPython([x.getSizes(), b.getSizes()], comm=da.comm)
        self.A.setPythonContext(self)
        
#        self.A = PETSc.Mat()
#        self.A.create(PETSc.COMM_WORLD)
#        self.A.setSizes([self.grid.nx*(self.grid.nv+1), self.grid.nx*(self.grid.nv+1)])
#        self.A.setType('aij') # sparse
#        self.A.setFromOptions()
        
        
        # create linear solver and preconditioner
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(A)
        self.ksp.setType('cg')
#        self.ksp.setType('gmres')
        
        self.pc = ksp.getPC()
        self.pc.setType('none')
        
        self.ksp.setFromOptions()
        
#        self.ksp.setType('preonly')
#        self.ksp.getPC().setType('lu')

        
        # create Arakawa matrix object
        self.arakawa = Arakawa(grid.nx, grid.nv)
        
        
        # create empty matrices for hstack and vstack operations
        self.E1 = csc_matrix((self.grid.nx * self.grid.nv, self.grid.nx * self.grid.nv))
        self.E2 = csc_matrix((self.grid.nx * self.grid.nv, self.grid.nx               ))
        self.E3 = csc_matrix((self.grid.nx,                self.grid.nx * self.grid.nv))
        self.E4 = csc_matrix((self.grid.nx,                self.grid.nx               ))
        
        
        # create matrix representations
        self.T = None
        self.D = None
        
        self.M  = None
        self.K  = None
        self.L  = None
        
        self.M0 = None
        self.K0 = None
        self.L0 = None
        
        self.M1 = None
        self.K1 = None
        self.L1 = None
        
        self.A0 = None
        self.A1 = None
        self.A2 = None
        
        self.B1 = None
        
    
    
    def calculate_constant_matrices(self, h, p):
        if self.T == None:
            Tvec    = 3. * np.ones(self.grid.nx * self.grid.nv) / self.ht
            Tarr    = np.array([Tvec, 2.*Tvec, Tvec])
            self.T  = create_periodic_band_matrix( self.grid.nx, self.grid.nv, Tarr, 2.*Tarr, Tarr, dirichlet=self.grid.dirichlet )
        
        
        if self.D == None:
            D = lil_matrix((self.grid.nx*self.grid.nv, self.grid.nx*self.grid.nv))
                    
            for i in range(0, self.grid.nx):
                D[    i*(self.grid.nv),       i*(self.grid.nv)  ] = 1. / self.grid.ht
                D[(i+1)*(self.grid.nv)-1, (i+1)*(self.grid.nv)-1] = 1. / self.grid.ht
                
            self.D = D.tocsc()
        
        
        if self.A0 == None:
            self.A0 = self.arakawa.create_arakawa_matrix(h.h0, dirichlet=self.grid.dirichlet) / (self.grid.hx * self.grid.hv)
        
        
        if self.M0 == None:
            if (self.grid.dirichlet):
                self.M0 = self.T - 2. * self.A0 + self.D
            else:
                self.M0 = self.T - 2. * self.A0
        
        if self.K0 == None:
            self.K0 = 4. * self.A0
        
        if self.L0 == None:
            self.L0 = self.T + 2. * self.A0
        
    
    def solve(self, f, h, p):
        nx = self.grid.nx
        nv = self.grid.nv
        
        # obtain sol & rhs vectors
        x, b = self.A.getVecs()
        
        
        self.calculate_constant_matrices(h, p)

        self.A1 = self.arakawa.create_arakawa_matrix(h.history1[:,:,0], dirichlet=self.grid.dirichlet) / (self.grid.hx * self.grid.hv)
        self.A2 = self.arakawa.create_arakawa_matrix(h.history1[:,:,1], dirichlet=self.grid.dirichlet) / (self.grid.hx * self.grid.hv)
        self.B1 = self.arakawa.create_inverse_matrix(f.history [:,:,0], dirichlet=self.grid.dirichlet) / (self.grid.hx * self.grid.hv)
        
        self.M1 = - 2.*self.A1
        self.K1 = + 2.*self.A2
        self.L1 = + 2.*self.A1
        
        self.M  = vstack([ hstack([self.M0 + self.M1, -2.*self.B1]), hstack([-1.*p.poisson.V, +1.*p.poisson.P]) ], format='csc')
#        self.K  = vstack([ hstack([self.K0 + self.K1,     self.E2]), hstack([+2.*p.poisson.V, -2.*p.poisson.P]) ], format='csc')
#        self.L  = vstack([ hstack([self.L0 + self.L1,     self.E2]), hstack([+1.*p.poisson.V, -1.*p.poisson.P]) ], format='csc')
        self.K  = vstack([ hstack([self.K0 + self.K1,     self.E2]), hstack([self.E3, self.E4]) ])
        self.L  = vstack([ hstack([self.L0 + self.L1,     self.E2]), hstack([self.E3, self.E4]) ])
        
        
        # concatenate f and phi for timesteps t and t-1
        u1 = np.concatenate( (f.history[:,:,0].ravel(), p.history[:,0]) )
        u2 = np.concatenate( (f.history[:,:,1].ravel(), p.history[:,1]) )
        
        # boundary conditions for Poisson equation
        # to subtract background density
        bc = np.zeros(nx*(nv+1))
        bc[nx*nv:] = p.poisson.boundary_conditions(f.history[:,:,0]) 
#        bc[nx*nv:] = 1. 
        if self.grid.dirichlet:
            bc[self.grid.nx*(self.grid.nv+1)-1] = 0.
        
        
#        np.savetxt('b.txt', self.K.dot(u1) + self.L.dot(u2) + bc, fmt='%12.4E')
#        np.savetxt('A.txt', self.M.todense(), fmt='%12.4E')
        
        
        # setup LHS matrix
        M_coo = self.M.tocoo()
        
#        self.A.setValues(M_coo.row, M_coo.col, M_coo.data)
        
        for i in range(0, len(M_coo.data)):
            self.A[M_coo.row[i], M_coo.col[i]] = M_coo.data[i]
        
        
        # communicate off-processor values
        # and setup internal data structures
        # for performing parallel operations
        self.A.assemblyBegin()
        self.A.assemblyEnd()
        
        
        self.formRHS(b)
        
        # calculate K.u1 and L.u2 and sum up the solution vector b
        b[:] = self.K.dot(u1) + self.L.dot(u2) + bc
        
        #
        x.set(0)
        
        # run actual solver
        self.ksp.setOperators(self.A)
        self.ksp.setFromOptions()
        self.ksp.solve(b, x)
        
        # update distribution function and potential from solution vector
        # !! # check if process id == 0 # !! #
        u = da.createNaturalVec()
        self.da.globalToNatural(x, u)
        
        f.update_distribution(u[0:nx*nv])
        p.update_potential(u[nx*nv:])
        
        # calculate residual
        residual         = abs( self.M.dot(x) - b ).max()
        poisson_residual = p.poisson.residual(p.phi, f.f)
            
        print("  solver residual = %24.16E, poisson derivative = %24.16E, poisson residual = %24.16E" % (residual, poisson_residual[0], poisson_residual[1]))
            
        
    def isSparse(self):
        return True
    
    
