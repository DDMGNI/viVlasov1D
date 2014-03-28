
import numpy as np

from vlasov.toolbox.maxwell import maxwellian


def distribution(x, v):
    f = (1. - 0.01 * np.cos(2. * np.pi * x)) * 8./3. * (np.sin(np.pi * v))**4
    
    return f 

