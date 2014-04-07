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
                 )
              ]
                
setup(
    name = 'PETSc Vlasov Solvers',
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
#     ext_modules = cythonize(extensions),
)
