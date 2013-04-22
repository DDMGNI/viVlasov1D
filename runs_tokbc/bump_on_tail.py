
import numpy as np

from vlasov.data.maxwell import maxwellian


def distribution(x, v):
    f  = 0.9 * maxwellian(1., v) + 0.2 * maxwellian(0.25, v, -4.5)
    f *= (1. + 0.03 * np.cos(0.3 * x))
    
    return f 
