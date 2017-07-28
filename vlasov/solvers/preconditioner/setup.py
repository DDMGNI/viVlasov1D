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
#         Extension("TensorProductDiagonal",
#                   sources=["TensorProductDiagonal.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
        Extension("TensorProductKinetic",
                  sources=["TensorProductKinetic.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("TensorProductKineticFast",
                  sources=["TensorProductKineticFast.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("TensorProductKineticSciPy",
                  sources=["TensorProductKineticSciPy.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("TensorProductPotential",
                  sources=["TensorProductPotential.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("TensorProductPotentialFast",
                  sources=["TensorProductPotentialFast.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("TensorProductPotentialSciPy",
                  sources=["TensorProductPotentialSciPy.pyx"],
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
