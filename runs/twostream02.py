
import numpy as np

from data.maxwell import maxwellian


def distribution(x, v):
    f = maxwellian(1., v) * (1. + 0.05 * np.cos(0.5 * x))
    
    if v != 0.:
        f *= v**2
    
    return f 

