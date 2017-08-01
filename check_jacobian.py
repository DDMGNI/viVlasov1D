'''
Created on June 06, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

import argparse

from run_base import viVlasov1Dbase

from vlasov.predictor.PETScArakawaRK4       import PETScArakawaRK4
from vlasov.predictor.PETScArakawaGear      import PETScArakawaGear

# from vlasov.vi.PETScNLArakawaJ1            import PETScSolver
# from vlasov.vi.PETScNLArakawaJ2            import PETScSolver
from vlasov.vi.PETScNLArakawaJ4            import PETScSolver

# from vlasov.predictor.PETScPoissonMatrixJ1     import PETScPoissonMatrix
# from vlasov.predictor.PETScPoissonMatrixJ2     import PETScPoissonMatrix
from vlasov.predictor.PETScPoissonMatrixJ4     import PETScPoissonMatrix


# solver_package = 'superlu_dist'
solver_package = 'mumps'
# solver_package = 'pastix'


class viVlasov1Djacobian(viVlasov1Dbase):
    '''
    PETSc/Python Vlasov Poisson Solver in 1D.
    '''


    def check_jacobian(self):
        
        (xs, xe), = self.da1.getRanges()
        
        eps = 1.E-7
        
        # update previous iteration
        self.petsc_solver.update_previous(self.x)
        
        # calculate jacobian
        self.petsc_solver.formJacobian(self.J)
        
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
    
    petscvp = viVlasov1Djacobian(args.runfile)
    petscvp.check_jacobian()
    
