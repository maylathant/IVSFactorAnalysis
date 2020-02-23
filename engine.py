import pandas as pd
from collections import defaultdict
import numpy as np
from statics import *
from utils import *

class VolSurf:
    '''
    Class to store and perform calculation on volatility surfaces
    '''
    def __init__(self,vpath,h1=1,h2=1):
        '''
        :param vpath: Path to the volatility file
        :param h1: Paramterter for the Nadaraya-Watson interpolation
        :param h2: Paramterter for the Nadaraya-Watson interpolation
        '''
        self.myvols = defaultdict(list)
        self.load_vols(vpath)
        self.h1 = h1
        self.h2 = h2

    def nwInterp(self,strike,mat,dat):
        '''
        Computes the Nadaraya-Watson interpolation for a point on the vol surface
        :param strike: Moneyness of the interp point
        :param mat: Maturity in years of the interp point
        :param dat: Date for which to reference for interpolation
        :return: Implied vol for a strike and maturity
        '''
        #Compute difference to reference
        matdiff = self.myvols[(dat,'matyears')] - mat
        strdiff = self.myvols[(dat, 'moneyness')] - strike
        denom = sum(gfunc(matdiff,strdiff,self.h1,self.h2))
        num = np.inner(self.myvols[(dat, 'normalizediv')],gfunc(matdiff,strdiff,self.h1,self.h2))
        return num/denom

    def load_vols(self,vpath):
        '''
        Read vols from a csv file
        :param vpath: Path to the csv file
        :return: Nothing, sets an instance variable
        '''
        raw = pd.read_csv(vpath).to_dict('records')
        full_dates = defaultdict(list)
        #histo = defaultdict(lambda: defaultdict(list)) #Reformat for historical calculations
        for ele in raw:
            for field in suf_map:
                full_dates[(ele['date'],field)].append(ele[field])
            #histo[ele['date']][(ele['moneyness'],ele['matyears'])].append(ele['normalizediv'])
        for ele in raw: #Transform into numpy array
            for field in suf_map:
                full_dates[(ele['date'],field)]= np.array(full_dates[(ele['date'],field)])
        # for ele in full_dates:
        #     full_dates[ele] = np.array(full_dates[ele])
        self.myvols = full_dates


if __name__ == '__main__':
    mySurf = VolSurf('data_download.csv',h1=0.5,h2=0.5)
    test = mySurf.nwInterp(1.055,0.1,'1/31/20')
    tim = np.linspace(0.05,4,30)
    money = np.linspace(0.5,1.5,30)
    tim, money = np.meshgrid(tim,money)
    vecfunc = np.vectorize(mySurf.nwInterp)
    ivs = vecfunc(money,tim,'1/31/20')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(tim,money,ivs)
    plt.show()
    print("Done")