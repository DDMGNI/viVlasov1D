#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from vlasov.setup_inc import *


ext_modules = [
        Extension("PETScVlasovSolver",
                  sources=["PETScVlasovSolver.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("PETScVlasovPreconditioner",
                  sources=["PETScVlasovPreconditioner.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
               
        Extension("PETScVlasovArakawaJ4",
                  sources=["PETScVlasovArakawaJ4.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        
        Extension("PETScNLVlasovArakawaJ1",
                  sources=["PETScNLVlasovArakawaJ1.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("PETScNLVlasovArakawaJ2",
                  sources=["PETScNLVlasovArakawaJ2.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("PETScNLVlasovArakawaJ4",
                  sources=["PETScNLVlasovArakawaJ4.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        
        Extension("PETScNLVlasovArakawaJ4kinetic",
                  sources=["PETScNLVlasovArakawaJ4kinetic.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("PETScNLVlasovArakawaJ4potential",
                  sources=["PETScNLVlasovArakawaJ4potential.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
               
        Extension("PETScNLVlasovArakawaJ4TensorFast",
                  sources=["PETScNLVlasovArakawaJ4TensorFast.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("PETScNLVlasovArakawaJ1TensorFast",
                  sources=["PETScNLVlasovArakawaJ1TensorFast.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
#         Extension("PETScNLVlasovArakawaJ4TensorPETSc",
#                   sources=["PETScNLVlasovArakawaJ4TensorPETSc.pyx"],
#                   include_dirs=INCLUDE_DIRS + [os.curdir],
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
        Extension("PETScNLVlasovArakawaJ4TensorSciPy",
                  sources=["PETScNLVlasovArakawaJ4TensorSciPy.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        
        Extension("PETScNLVlasovArakawaJ4RK2",
                  sources=["PETScNLVlasovArakawaJ4RK2.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("PETScNLVlasovArakawaJ4RK4",
                  sources=["PETScNLVlasovArakawaJ4RK4.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        
        Extension("PETScNLVlasovSimpson",
                  sources=["PETScNLVlasovSimpson.pyx"],
                  include_dirs=INCLUDE_DIRS + [os.curdir],
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 )
        
              ]
                
setup(
    name = 'PETSc Vlasov Solver',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
