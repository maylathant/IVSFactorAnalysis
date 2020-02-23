import numpy as np
from scipy.stats import norm

def gfunc(m,t,h1,h2):
    '''
    Function to compute interpolation helper (Similar to Radial Basis)
    :param m: Difference
    :return:Smoothed real number
    '''
    return norm.pdf(m,h1)*norm.pdf(t,h2)