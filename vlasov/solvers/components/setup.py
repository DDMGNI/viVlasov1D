#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from vlasov.setup_inc import *


ext_modules = [
        Extension("Collisions",
                  sources=["Collisions.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("PoissonBracket",
                  sources=["PoissonBracket.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("Regularisation",
                  sources=["Regularisation.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 ),
        Extension("TimeDerivative",
                  sources=["TimeDerivative.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS
                 )
              ]

setup(
    name = 'PETSc Variational Vlasov-Poisson Solver',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
