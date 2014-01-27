#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from vlasov.setup_inc import *


ext_modules = [
        Extension("PETScPoissonSolver2",
                  sources=["PETScPoissonSolver2.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PETScPoissonSolver4",
                  sources=["PETScPoissonSolver4.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 )
              ]
                
setup(
    name = 'PETSc Vlasov-Poisson Explicit Predictor',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
