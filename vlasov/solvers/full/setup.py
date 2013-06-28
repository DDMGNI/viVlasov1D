#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Distutils    import build_ext

import os
from os.path import join, isdir

INCLUDE_DIRS = []
LIBRARY_DIRS = []
LIBRARIES    = []

# PETSc
PETSC_DIR  = os.environ['PETSC_DIR']
PETSC_ARCH = os.environ.get('PETSC_ARCH', '')

if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
    INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                     join(PETSC_DIR, 'include')]
    LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
else:
    if PETSC_ARCH: pass # XXX should warn ...
    INCLUDE_DIRS += [join(PETSC_DIR, 'include')]
    LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]

LIBRARIES    += ['petsc']

# NumPy
import numpy
INCLUDE_DIRS += [numpy.get_include()]

# PETSc for Python
import petsc4py
INCLUDE_DIRS += [petsc4py.get_include()]

# OpenMPI
INCLUDE_DIRS += ['/opt/local/include/openmpi']

# Intel MPI
INCLUDE_DIRS += ['/afs/@cell/common/soft/intel/impi/4.1.0/intel64/include']

# Valgrind
INCLUDE_DIRS += ['/opt/local/include']
LIBRARY_DIRS += ['/opt/local/lib']


ext_modules = [
        Extension("PETScFullSolver",
                  sources=["PETScFullSolver.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
               
        Extension("PETScArakawaJ1",
                  sources=["PETScArakawaJ1.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PETScArakawaJ2",
                  sources=["PETScArakawaJ2.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PETScArakawaJ4",
                  sources=["PETScArakawaJ4.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
               
        Extension("PETScNLArakawaJ1",
                  sources=["PETScNLArakawaJ1.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PETScNLArakawaJ2",
                  sources=["PETScNLArakawaJ2.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PETScNLArakawaJ4",
                  sources=["PETScNLArakawaJ4.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 )
              ]
                
setup(
    name = 'PETSc Variational Integrators for the Vlasov-Poisson System in 1D1V',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
