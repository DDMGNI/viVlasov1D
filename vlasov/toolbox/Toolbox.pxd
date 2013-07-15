'''
Created on Jan 25, 2013

@author: mkraus
'''

cimport cython
cimport numpy as np

from VIDA cimport VIDA

from petsc4py.PETSc cimport Vec


cdef class Toolbox(object):

    cdef np.uint64_t  nx
    cdef np.uint64_t  nv
    
    cdef np.float64_t ht
    cdef np.float64_t hx
    cdef np.float64_t hv
    
    cdef np.float64_t ht_inv
    cdef np.float64_t hx_inv
    cdef np.float64_t hv_inv
    
    cdef np.float64_t hx2
    cdef np.float64_t hv2
    cdef np.float64_t hx2_inv
    cdef np.float64_t hv2_inv
    
    cdef np.ndarray v
    
    cdef VIDA dax
    cdef VIDA da1
    

    cpdef potential_to_hamiltonian(self, Vec P, Vec H)
    cpdef compute_density(self, Vec F, Vec N)
    cpdef compute_velocity_density(self, Vec F, Vec U)
    cpdef compute_energy_density(self, Vec F, Vec E)
    cpdef compute_collision_factor(self, Vec N, Vec U, Vec E, Vec A)

    cdef compute_density_array(self, np.ndarray[np.float64_t, ndim=2] f, np.ndarray[np.float64_t, ndim=1] n)
    cdef compute_velocity_density_array(self, np.ndarray[np.float64_t, ndim=2] f, np.ndarray[np.float64_t, ndim=1] u)
    cdef compute_energy_density_array(self, np.ndarray[np.float64_t, ndim=2] f, np.ndarray[np.float64_t, ndim=1] e)
    cdef compute_collision_factor_array(self, np.ndarray[np.float64_t, ndim=1] n,
                                              np.ndarray[np.float64_t, ndim=1] u,
                                              np.ndarray[np.float64_t, ndim=1] e,
                                              np.ndarray[np.float64_t, ndim=1] a)

#     cdef maxwellian(self, np.float64_t temperature, np.float64_t velocity, np.float64_t vOffset)
#     cdef boltzmannian(self, np.float64_t temperature, np.float64_t energy)
