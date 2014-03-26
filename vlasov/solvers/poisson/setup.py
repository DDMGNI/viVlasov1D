#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from vlasov.setup_inc import *


ext_modules = [
        Extension("PETScPoissonSolver",
                  sources=["PETScPoissonSolver.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PETScPoissonCFD2",
                  sources=["PETScPoissonCFD2.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PETScPoissonCFD4",
                  sources=["PETScPoissonCFD4.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PETScPoissonSimpson",
                  sources=["PETScPoissonSimpson.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 )
              ]
                
setup(
    name = 'PETSc Vlasov-Poisson Poisson Schemes',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
