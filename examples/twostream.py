"""
Twostream Instability

   Initial distribution function:
   
   :math:`f(x,v) = v^2 \, f_{M} \, ( 1 + A \, \cos{( k_{x} \, x) } )`
   with :math:`A = 0.05, k = 0.5`
"""

import numpy as np

from vlasov.toolbox.maxwell import maxwellian


def distribution(x, v):
    f = maxwellian(1., v) * (1. + 0.05 * np.cos(0.5 * x)) * v**2
    
    return f 

