#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from vlasov.setup_inc import *


ext_modules = [
        Extension("TensorProduct",
                  sources=["TensorProduct.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("TensorProductFast",
                  sources=["TensorProductFast.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("TensorProductSciPy",
                  sources=["TensorProductSciPy.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 )
              ]

setup(
    name = 'PETSc Variational Vlasov-Poisson Solver',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
