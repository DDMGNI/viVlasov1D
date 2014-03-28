#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from vlasov.setup_inc import *


ext_modules = [
              ]

setup(
    name = 'PETSc Variational Vlasov-Poisson Solver',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
