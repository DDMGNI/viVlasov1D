'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.core.Grid    cimport Grid
from vlasov.toolbox.VIDA cimport VIDA

from vlasov.solvers.components.Collisions     cimport Collisions
from vlasov.solvers.components.PoissonBracket cimport PoissonBracket
from vlasov.solvers.components.Regularisation cimport Regularisation
from vlasov.solvers.components.TimeDerivative cimport TimeDerivative


# ctypedef void (*f_poisson_bracket)(Vec, Vec, Vec, double)
# ctypedef void (*f_time_derivative)(Vec, Vec)
# ctypedef void (*f_collision_operator)(Vec, Vec, Vec, Vec, Vec, Vec, double)
# ctypedef void (*f_regularisation)(Vec, Vec, double)


cdef class PETScVlasovSolverBase(object):

    cdef PoissonBracket poisson_bracket
    cdef TimeDerivative time_derivative
    cdef Regularisation regularisation
    cdef Collisions     collisions
    
#     cdef f_poisson_bracket    poisson_bracket_function
#     cdef f_time_derivative    time_derivative_function
#     cdef f_collision_operator collision_operator_function
#     cdef f_regularisation     regularisation_function
    
    cdef double charge
    
    cdef VIDA da1
    cdef Grid grid
    
    cdef Vec H0
    cdef Vec H1p
    cdef Vec H1h
    cdef Vec H2p
    cdef Vec H2h
    
    cdef Vec Np
    cdef Vec Up
    cdef Vec Ep
    cdef Vec Ap
    
    cdef Vec Nh
    cdef Vec Uh
    cdef Vec Eh
    cdef Vec Ah
    
    cdef Vec Fave
    cdef Vec Fder
    cdef Vec Have
    
    cdef Vec Fp
    cdef Vec Fh
    
    cdef Vec localFave
    cdef Vec localFder
    cdef Vec localHave
    
    cdef Vec localFp
    cdef Vec localFh
    cdef Vec localFd
    
    
    cpdef jacobian(self, Vec F, Vec Y)
    cpdef function(self, Vec F, Vec Y)
    
    cpdef mult(self, Mat mat, Vec F, Vec Y)
    cpdef snes_mult(self, SNES snes, Vec F, Vec Y)
    cpdef jacobian_mult(self, Vec F, Vec Y)
    
    cpdef function_snes_mult(self, SNES snes, Vec F, Vec Y)
    cpdef function_mult(self, Vec F, Vec Y)

    cdef jacobian_solver(self, Vec F, Vec Y)
    cdef function_solver(self, Vec F, Vec Y)
    
    cdef call_poisson_bracket(self, Vec F, Vec H, Vec Y, double factor)
    cdef call_time_derivative(self, Vec F, Vec Y)
    cdef call_collision_operator(self, Vec F, Vec Y, Vec N, Vec U, Vec E, Vec A, double factor)
    cdef call_regularisation(self, Vec F, Vec Y, double factor)
    
