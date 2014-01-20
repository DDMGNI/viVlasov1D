'''
Created on June 05, 2013

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport numpy as npy

from petsc4py.PETSc cimport SNES, Mat, Vec

from vlasov.toolbox.Grid    cimport Grid
from vlasov.toolbox.VIDA    cimport VIDA
from vlasov.toolbox.Toolbox cimport Toolbox


cdef class PETScFullSolverBase(object):

    cdef npy.float64_t charge
    cdef npy.float64_t nu
    cdef npy.float64_t coll_diff
    cdef npy.float64_t coll_drag
    cdef npy.float64_t regularisation
    
    cdef VIDA dax
    cdef VIDA da1
    cdef VIDA da2
    cdef Grid grid
    
    cdef Vec H0
    cdef Vec H1p
    cdef Vec H1h
    cdef Vec H1d
    cdef Vec H2p
    cdef Vec H2h
    
    cdef Vec Fp
    cdef Vec Pp
    cdef Vec Np
    cdef Vec Up
    cdef Vec Ep
    cdef Vec Ap
    
    cdef Vec Fh
    cdef Vec Ph
    cdef Vec Nh
    cdef Vec Uh
    cdef Vec Eh
    cdef Vec Ah
    
    cdef Vec Fd
    cdef Vec Pd
    cdef Vec Nd
    cdef Vec Ud
    cdef Vec Ed
    cdef Vec Ad
    
    cdef Vec Nc
    cdef Vec Uc
    cdef Vec Ec
    
    cdef Vec localH0
    cdef Vec localH1p
    cdef Vec localH1h
    cdef Vec localH1d
    cdef Vec localH2p
    cdef Vec localH2h
    
    cdef Vec localFp
    cdef Vec localPp
    cdef Vec localNp
    cdef Vec localUp
    cdef Vec localEp
    cdef Vec localAp
    
    cdef Vec localFh
    cdef Vec localPh
    cdef Vec localNh
    cdef Vec localUh
    cdef Vec localEh
    cdef Vec localAh
    
    cdef Vec localFd
    cdef Vec localPd
    cdef Vec localNd
    cdef Vec localUd
    cdef Vec localEd
    cdef Vec localAd
    
    cdef Vec localNc
    cdef Vec localUc
    cdef Vec localEc
    
    cdef npy.ndarray h0
    cdef npy.ndarray h1p
    cdef npy.ndarray h1h
    cdef npy.ndarray h1d
    cdef npy.ndarray h2p
    cdef npy.ndarray h2h
    
    cdef npy.ndarray fp
    cdef npy.ndarray pp
    cdef npy.ndarray np
    cdef npy.ndarray up
    cdef npy.ndarray ep
    cdef npy.ndarray ap
    
    cdef npy.ndarray fh
    cdef npy.ndarray ph
    cdef npy.ndarray nh
    cdef npy.ndarray uh
    cdef npy.ndarray eh
    cdef npy.ndarray ah
    
    cdef npy.ndarray fd
    cdef npy.ndarray pd
    cdef npy.ndarray nd
    cdef npy.ndarray ud
    cdef npy.ndarray ed
    cdef npy.ndarray ad
    
    cdef npy.ndarray nc
    cdef npy.ndarray uc
    cdef npy.ndarray ec
    
    cdef Toolbox toolbox
    
    
    cdef get_data_arrays(self)
