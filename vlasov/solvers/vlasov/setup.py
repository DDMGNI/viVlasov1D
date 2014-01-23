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
#FLAGS        = []
FLAGS        = ["-axavx"]

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
        Extension("PETScVlasovSolver",
                  sources=["PETScVlasovSolver.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
               
        Extension("PETScVlasovArakawaJ4",
                  sources=["PETScVlasovArakawaJ4.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
        
        Extension("PETScNLVlasovArakawaJ1",
                  sources=["PETScNLVlasovArakawaJ1.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
        Extension("PETScNLVlasovArakawaJ2",
                  sources=["PETScNLVlasovArakawaJ2.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
        Extension("PETScNLVlasovArakawaJ4",
                  sources=["PETScNLVlasovArakawaJ4.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
        
        Extension("PETScNLVlasovArakawaJ4kinetic",
                  sources=["PETScNLVlasovArakawaJ4kinetic.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
        Extension("PETScNLVlasovArakawaJ4potential",
                  sources=["PETScNLVlasovArakawaJ4potential.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
               
        Extension("PETScNLVlasovArakawaJ4TensorPETSc",
                  sources=["PETScNLVlasovArakawaJ4TensorPETSc.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
        Extension("PETScNLVlasovArakawaJ4TensorSciPy",
                  sources=["PETScNLVlasovArakawaJ4TensorSciPy.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
        
        Extension("PETScNLVlasovArakawaJ4RK2",
                  sources=["PETScNLVlasovArakawaJ4RK2.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
        Extension("PETScNLVlasovArakawaJ4RK4",
                  sources=["PETScNLVlasovArakawaJ4RK4.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 ),
        
        Extension("PETScNLVlasovSimpson",
                  sources=["PETScNLVlasovSimpson.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=FLAGS
                 )
        
              ]
                
setup(
    name = 'PETSc Vlasov Solver',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
