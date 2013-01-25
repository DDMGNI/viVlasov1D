
import numpy as np

t0    = 0.
tL    = 69.
tR    = 307.
tO    = 20.
k     = 0.26
omega = 0.37
Emax  = 0.2


def external(x, t):
    
    eps =   0.5 * ( np.tanh( (t0-tL)/tO ) - np.tanh( (t0-tR)/tO ) )
    a   = ( 0.5 * ( np.tanh( (t -tL)/tO ) - np.tanh( (t -tR)/tO ) ) - eps ) / (1.-eps) 
    
    potential = Emax * a * np.cos(k*x - omega*t)
    
    return potential
