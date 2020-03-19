import numpy as np
from scipy.stats import norm

def gfunc(m,t,h1,h2):
    '''
    Function to compute interpolation helper (Similar to Radial Basis)
    :param m: Difference in moneyness
    :param t: Differnce in strikes
    :return:Smoothed real number
    '''
    return 0.5/np.pi*np.exp(-m*m/2/h1)*np.exp(-t*t/2/h2)