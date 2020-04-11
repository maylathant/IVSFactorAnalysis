import pandas as pd
from collections import defaultdict
import numpy as np
from statics import *
from utils import *
from sklearn.decomposition import PCA
import numpy.polynomial as npbasis
from plots import *
from operator import itemgetter

class Basis:
    '''
    Class to handle basis functions
    '''
    def __init__(self,balen):
        '''
        :param balen: Max number of basis functions to use in the sequence
        '''
        if np.sqrt(balen) - round(np.sqrt(balen),0) > 0:
            raise ValueError("Basis Truncation Length Must be a Perfect Square")
        self.balen = balen
        self.badim = np.sqrt(balen)

    def get_one_bas(self,n):
        '''
        Get the nth basis function
        :param n: index for basis function
        :return: nth basis function
        '''
        if n > self.balen:
            raise ValueError("Function index is larger than maximum")
        bacof = np.zeros(self.balen)
        bacof[n-1] = 1
        bacof = bacof.reshape((self.badim,self.badim))
        return lambda x,y : npbasis.laguerre.lagval2d(x,y,bacof)

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
        self.diff_dates = defaultdict(list)
        self.dates = []
        self.contracts = []
        self.maxcont = {}
        self.mincont = {}
        self.load_vols(vpath)
        self.h1 = h1
        self.h2 = h2
        self.pca = None


    def nwInterp(self,strike,mat,dat,diff=False):
        '''
        Computes the Nadaraya-Watson interpolation for a point on the vol surface
        :param strike: Moneyness of the interp point
        :param mat: Maturity in years of the interp point
        :param dat: Date for which to reference for interpolation
        :param diff: If true, use vol % differences instead of volatility quote
        :return: Implied vol for a strike and maturity
        '''
        interp_vol = self.diff_dates if diff else self.myvols
        #Compute difference to reference
        matdiff = interp_vol[(dat,'matyears')] - mat
        strdiff = interp_vol[(dat, 'moneyness')] - strike
        denom = sum(gfunc(matdiff,strdiff,self.h1,self.h2))
        num = np.inner(interp_vol[(dat, 'normalizediv')],gfunc(matdiff,strdiff,self.h1,self.h2))
        return num/denom

    def load_vols(self,vpath):
        '''
        Read vols from a csv file
        :param vpath: Path to the csv file
        :return: Nothing, sets an instance variable
        '''
        raw = pd.read_csv(vpath).sort_values(by=['date'], ascending=True)
        tempmin = raw.min(); tempmax = raw.max()
        for s in suf_map:
            self.mincont[s] = tempmin[s]
            self.maxcont[s] = tempmax[s]
        mask = (raw['call/put'] != 'C') | (raw['moneyness'] != 1)
        self.dates = list(raw['date'].unique())
        raw = raw[mask].to_dict('records')
        full_dates = defaultdict(list)
        histo = defaultdict(list)
        diff_dates = defaultdict(list)
        for ele in raw:
            for field in suf_map:
                full_dates[(ele['date'],field)].append(ele[field])
            histo[(ele['moneyness'],ele['matyears'])].append(ele['normalizediv'])
        self.contracts = list(histo.keys())
        mysurface = np.empty((len(self.dates),))
        for ele in histo:
            mysurface = np.vstack([mysurface,histo[ele]])
        temp_dates = mysurface[1:] #Format changes in surface
        temp_dates = np.diff(temp_dates, axis=1) / temp_dates[:, :-1]
        for date in range(len(self.dates)-1):
            for cont in range(len(self.contracts)):
                diff_dates[(self.dates[date], 'moneyness')].append(self.contracts[cont][0])
                diff_dates[(self.dates[date],'matyears')].append(self.contracts[cont][1])
                diff_dates[(self.dates[date],'normalizediv')].append(temp_dates[cont][date])
        #Change format to numpy array
        for ele in full_dates:
            full_dates[ele] = np.array(full_dates[ele])
            diff_dates[ele] = np.array(diff_dates[ele])
        self.myvols = full_dates
        self.histvols = mysurface[1:]
        self.diff_dates = diff_dates

    def get_pca(self,n : int):
        '''
        Compute an sklean pca object for the historical vol surface
        :param n: (int) Number of factors in PCA
        :return: None
        '''
        pca = PCA(n_components=n)
        tempvols = np.diff(self.histvols,axis=1)/self.histvols[:,:-1]
        tempvols = (tempvols-np.mean(tempvols,axis=0))/np.std(tempvols,axis=0)
        tempvols = np.cov(tempvols)
        pca.fit(np.transpose(tempvols))
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

    def proj_fact(self):
        '''
        Project PCA factors onto a time series of implied vols
        :return: An nxt array where n is the number of PCA factors and t
        is the number of time series points
        '''
        if self.pca is None:
            print('Error: Must Run get_pca() before proj_fact()')
            exit(1)
        return np.matmul(self.pca.components_,mySurf.histvols)



if __name__ == '__main__':
    print('Welcome to the Volatility Surfice PCA Demo!')
    print('You will see surface plots of the most significant volatility factors in 2019.\n'
          ' These will appear in Sepereate Windows.\n')
    input("Press Enter to continue...")
    print('Computing...')
    mySurf = VolSurf('data_download.csv',h1=0.1,h2=0.1)
    pca_factors = 4; graph_scn = 'pca3'
    mySurf.get_pca(pca_factors)
    mySurf.map_pca()
    print('Rendering Graphs...')
    for i in range(pca_factors): #Plot surfaces of volatility factors
        graph_scn = 'pca' + str(i)
        plt_surf(mySurf,graph_scn,diff=False,save=False,title='Volatility Plot for Factor ' + str(i+1))

    plt_importance(mySurf) #Plot importance of each factor

    plt_proj_time(mySurf,pca_factors) #Plot time series of factor magnitude

    print("Done!")