
import numpy as np

from vlasov.data.maxwell import maxwellian


def distribution(x, v):
    f = maxwellian(1., v) * (1. + 0.05 * np.cos(0.5 * x)) * v**2
    
    return f 

