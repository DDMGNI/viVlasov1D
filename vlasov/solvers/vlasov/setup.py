#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from vlasov.setup_inc import *


# extensions = [Extension("*", ["*.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS),
#               ]
                
extensions = [
        Extension("PETScVlasovSolver",
                  sources=["PETScVlasovSolver.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
              
        Extension("PETScNLVlasovMP",
                  sources=["PETScNLVlasovMP.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
               
        Extension("PETScNLVlasovTensorFastMP",
                  sources=["PETScNLVlasovTensorFastMP.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
               
#         Extension("PETScNLVlasovArakawaJ1DB2",
#                   sources=["PETScNLVlasovArakawaJ1DB2.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
#         Extension("PETScNLVlasovArakawaJ1DBf",
#                   sources=["PETScNLVlasovArakawaJ1DBf.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
#         Extension("PETScNLVlasovArakawaJ1DBh",
#                   sources=["PETScNLVlasovArakawaJ1DBh.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
              
#         Extension("PETScNLVlasovArakawaJ4kinetic",
#                   sources=["PETScNLVlasovArakawaJ4kinetic.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
#         Extension("PETScNLVlasovArakawaJ4potential",
#                   sources=["PETScNLVlasovArakawaJ4potential.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
               
#         Extension("PETScNLVlasovArakawaJ1TensorFast",
#                   sources=["PETScNLVlasovArakawaJ1TensorFast.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
#         Extension("PETScNLVlasovArakawaJ1AveMTensorFast",
#                   sources=["PETScNLVlasovArakawaJ1AveMTensorFast.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
#         Extension("PETScNLVlasovArakawaJ1AveSTensorFast",
#                   sources=["PETScNLVlasovArakawaJ1AveSTensorFast.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
#         Extension("PETScNLVlasovArakawaJ4TensorFast",
#                   sources=["PETScNLVlasovArakawaJ4TensorFast.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
#         Extension("PETScNLVlasovArakawaJ4TensorSciPy",
#                   sources=["PETScNLVlasovArakawaJ4TensorSciPy.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
#         Extension("PETScNLVlasovArakawaJ4AveATensorFast",
#                   sources=["PETScNLVlasovArakawaJ4AveATensorFast.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
#         Extension("PETScNLVlasovArakawaJ4AveSTensorFast",
#                   sources=["PETScNLVlasovArakawaJ4AveSTensorFast.pyx"],
#                   include_dirs=INCLUDE_DIRS,
#                   libraries=LIBRARIES,
#                   library_dirs=LIBRARY_DIRS,
#                   runtime_library_dirs=LIBRARY_DIRS,
#                   extra_compile_args=CARGS,
#                   extra_link_args=LARGS
#                  ),
        
        Extension("PETScNLVlasovArakawaJ4RK2",
                  sources=["PETScNLVlasovArakawaJ4RK2.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        Extension("PETScNLVlasovArakawaJ4RK4",
                  sources=["PETScNLVlasovArakawaJ4RK4.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 ),
        
        Extension("PETScNLVlasovSimpson",
                  sources=["PETScNLVlasovSimpson.pyx"],
                  include_dirs=INCLUDE_DIRS,
                  libraries=LIBRARIES,
                  library_dirs=LIBRARY_DIRS,
                  runtime_library_dirs=LIBRARY_DIRS,
                  extra_compile_args=CARGS,
                  extra_link_args=LARGS
                 )
        
              ]
                
setup(
    name = 'PETSc Vlasov Solvers',
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
#     ext_modules = cythonize(extensions),
)
