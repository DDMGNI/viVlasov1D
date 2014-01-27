'''
Created on 27.01.2014

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Distutils    import build_ext

import numpy
import os
from os.path import join, isdir

INCLUDE_DIRS = []
LIBRARY_DIRS = []
LIBRARIES    = []
CARGS        = ['-O3','-axavx', '-std=c99']
LARGS        = []

# FFTW
FFTW_DIR  = os.environ['FFTW_HOME']

if FFTW_DIR:
    INCLUDE_DIRS += [join(FFTW_DIR, 'include')]
    LIBRARY_DIRS += [join(FFTW_DIR, 'lib')]
    
    LIBRARIES    += ['fftw3']


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

# LAPACK
if 'extra_compile_args' in numpy.__config__.lapack_opt_info:
    CARGS += numpy.__config__.lapack_opt_info['extra_compile_args']

if 'extra_link_args' in numpy.__config__.lapack_opt_info:
    LARGS += numpy.__config__.lapack_opt_info['extra_link_args']

if 'include_dirs' in numpy.__config__.lapack_opt_info:
    INCLUDE_DIRS += numpy.__config__.lapack_opt_info['include_dirs']

if 'library_dirs' in numpy.__config__.lapack_opt_info:
    LIBRARY_DIRS += numpy.__config__.lapack_opt_info['library_dirs']

if 'libraries' in numpy.__config__.lapack_opt_info:
    LIBRARIES += numpy.__config__.lapack_opt_info['libraries']

