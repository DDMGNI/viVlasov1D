
from numpy import cos, exp, pi, sqrt

eps = 0.01
k   = 0.2
v0  = 2.4


def distribution(x, v):
    return 0.5 / sqrt(2. * pi) * (1. + eps * cos(k * x)) * ( exp( - (v-v0)**2 ) + exp( - (v+v0)**2 ) )

