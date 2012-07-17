'''
Created on Mar 29, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''


import numpy as np
from numpy.linalg import cond, inv, norm, solve
from scipy.sparse import csc_matrix, dok_matrix, eye, hstack, lil_matrix, spdiags, vstack
from scipy.sparse.linalg import eigs, spsolve

#import matplotlib.pyplot as plt


class Poisson():
    '''
    classdocs
    '''


    def __init__(self, grid, const=1.0):
        pass
    