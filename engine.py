import pandas as pd
from collections import defaultdict
import numpy as np
from statics import *
from utils import *
from sklearn.decomposition import PCA

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
        self.histvols = defaultdict(list)
        self.dates = []
        self.contracts = []
        self.load_vols(vpath)
        self.h1 = h1
        self.h2 = h2
        self.pca = None

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
        raw = pd.read_csv(vpath).sort_values(by=['date'])
        mask = (raw['call/put'] != 'C') | (raw['moneyness'] != 1)
        self.dates = list(raw['date'].unique())
        raw = raw[mask].to_dict('records')
        full_dates = defaultdict(list)
        histo = defaultdict(list)
        for ele in raw:
            for field in suf_map:
                full_dates[(ele['date'],field)].append(ele[field])
            histo[(ele['moneyness'],ele['matyears'])].append(ele['normalizediv'])
        #Change format to numpy array
        for ele in full_dates:
            full_dates[ele] = np.array(full_dates[ele])
        self.contracts = list(histo.keys())
        mysurface = np.empty((len(self.dates),))
        for ele in histo:
            mysurface = np.vstack([mysurface,histo[ele]])
        self.myvols = full_dates
        self.histvols = mysurface[1:]

    def get_pca(self,n : int):
        '''
        Compute an sklean pca object for the historical vol surface
        :param n: (int) Number of factors in PCA
        :return: None
        '''
        pca = PCA(n_components=n)
        tempvols = np.transpose(self.histvols)
        tempvols = np.diff(tempvols,axis=0)/tempvols[1:]
        pca.fit(tempvols)
        self.pca = pca
        return None

    def map_pca(self):
        '''
        Maps an existing PCA scenario to the myvols instance variable
        :return: None
        '''
        if self.pca is None:
            print('Error: Must Run get_pca() before map_pca()')
            exit(1)
        for n in range(self.pca.n_components_):
            for i in range(len(self.pca.components_[n])):
                self.myvols[('pca' + str(n),'moneyness')].append(self.contracts[i][0])
                self.myvols[('pca' + str(n), 'matyears')].append(self.contracts[i][1])
                self.myvols[('pca' + str(n), 'normalizediv')].append(self.pca.components_[n][i])
        for n in range(self.pca.n_components_):
            for field in suf_map:
                self.myvols[('pca' + str(n), field)] \
                    = np.array(self.myvols[('pca' + str(n), field)])



if __name__ == '__main__':
    pca_factors = 4; graph_scn = 'pca2'
    mySurf = VolSurf('data_download.csv',h1=0.5,h2=0.5)
    mySurf.get_pca(pca_factors)
    mySurf.map_pca()
    tim = np.linspace(0.05,4,30)
    money = np.linspace(0.5,1.5,30)
    tim, money = np.meshgrid(tim,money)
    vecfunc = np.vectorize(mySurf.nwInterp)
    ivs = vecfunc(money,tim,graph_scn)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(tim,money,ivs)
    plt.plot()
    print("Done")