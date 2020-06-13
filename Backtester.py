from engine import *
import os
import matplotlib.pyplot as plt


class Backtester:
    '''
    Back test PNL attribution with PCA factors
    '''
    cont_fn = "cont_w.p"
    rand_fn = "rands.p"

    def __init__(self,mySurf,refresh=False):
        '''
        :param mySurf: VolSurf object (required)
        '''
        self.mySurf = mySurf
        self.myrands = None
        self.mycontw = None
        self.get_contw(refresh)
        self.get_rands(refresh)
        self.pos = None
        self.btdone = False
        self.invest = None

    def cache_rand(mySurf):
        '''
        Cache normal random variables for simulation in pickle object
        :param mySurf: A volSurf object with histvols value initialized
        :return: Data that was cached
        '''
        rands = np.random.normal(0, 1, mySurf.histvols.shape)
        flname = Backtester.rand_fn
        pickle.dump(rands, open(flname, "wb"))
        return rands

    def cache_cont(mySurf):
        '''
        Cache normal random variables for simulation in pickle object
        :param mySurf: A volSurf object with histvols value initialized
        :return: Data that was cached
        '''
        conts = np.array([Backtester.contract_w(c) for c in mySurf.contracts])
        flname = Backtester.cont_fn
        pickle.dump(conts, open(flname, "wb"))
        return conts

    def contract_w(cont):
        '''
        Compute contract weight for backtesting
        :param cont: tuple containing (moneyness, years to maturity)
        :return: (float) contract weight
        '''
        M = cont[0]
        T = cont[1]
        money_fact = 2 - (M - 1) * (M - 1) + 1/M
        #money_fact = np.exp(-15*(M-1)*(M-1))
        return money_fact / np.power(T,1/3) / 36

    def get_rands(self, refresh):
        '''
        Gets random variables used in the simulation
        :param refresh: Bool. Refresh cache or not
        :return: None. Sets instance variable
        '''
        if os.path.exists(Backtester.rand_fn) and not refresh:
            self.myrands = pickle.load( open(Backtester.rand_fn, "rb" ) )
        else:
            self.myrands = Backtester.cache_rand(mySurf)
        return None

    def get_contw(self, refresh):
        '''
        Gets contract weights used in the simulation
        :param refresh: Bool. Refresh cache or not
        :return: None. Sets instance variable
        '''
        if os.path.exists(Backtester.cont_fn) and not refresh:
            self.mycontw = pickle.load( open( Backtester.cont_fn, "rb" ) )
        else:
            self.mycontw = Backtester.cache_cont(mySurf)
        return None

    def init_pos(self, invest, random=True):
        '''
        Initialize backtesting positions
        :param invest: Initial amount invested (float)
        :param random: If false, positions remain constant throughout; otherwise, random each day
        :return: None. Sets instance variable
        '''
        self.pos = self.myrands.copy()
        if not random:
            self.pos[:,0] = 1
            self.pos[:, 1:] = 0
        self.pos[:, 0] = self.pos[:, 0]* myBT.mycontw
        self.pos[:, 0] = invest * self.pos[:, 0] / sum(self.pos[:, 0])
        self.invest = invest
        return None

    def run_bt(self):
        '''
        Runs the backtest
        :return: None. Changes position instance variable
        '''
        if self.pos is None:
            raise ValueError('Please initialize positions before running backtest!')
        sim_days = np.shape(self.pos)[1]
        limit = self.invest/4
        for day in range(1,sim_days):
            self.pos[:, day] = self.pos[:,day-1]*(1 + self.mycontw*self.pos[:,day])
            self.pos[:, day][self.pos[:, day] > limit] = limit #Limit size of long positions
            self.pos[:, day][self.pos[:, day] < -limit] = -limit #Limit size of shorts
        self.btdone = True
        return None

    def get_pnl(self, factor=False , **kwargs):
        '''
        Computes PNL from backtesting results
        :param factor: Bool. If true, factor pnl; otherwise, total pnl
        :kwargs param shifts: nparray a vector of shifts for a factor w
        :kwargs param pc: nparray a principal factor
        :return: None. Changes instance variable
        '''
        if not self.btdone:
            raise ValueError('Please run backtest before computing PNL!')
        if self.mySurf.pca is None:
            raise ValueError('Please run PCA for VolSurface First!')
        pnl = np.diff(self.mySurf.histvols,axis=1)
        for i in range(len(pnl[0])):
            if factor:
                pnl[:,i] = self.pos[:,i]*kwargs['shifts'][i]*kwargs['pc']
            else:
                pnl[:, i] = self.pos[:, i]*pnl[:, i]
        return pnl


if __name__ == "__main__":
    investment = 1e6; random = True

    #Initialize vol surface
    mySurf = VolSurf('data_download.csv',h1=0.1,h2=0.1)

    # Run backtest
    myBT = Backtester(mySurf,refresh=True)
    myBT.init_pos(invest=investment, random=random)
    myBT.run_bt()
    # pickle.dump(myBT, open('myBT.p', "wb"))
    # myBT = pickle.load(open( 'myBT.p', "rb" ))

    #Compute PNL
    pca_factors = 188;
    myBT.mySurf.get_pca(pca_factors, cov=False)
    fact_impact = myBT.mySurf.proj_fact(diff=True)
    pnl = []
    pnl.extend([myBT.get_pnl(factor=False)])
    for i in range(pca_factors):
        pnl.extend([myBT.get_pnl(factor=True,shifts=fact_impact[i], pc=myBT.mySurf.pca.components_[i])])



    #Compute cumsum for each pnl
    pnlcs = []
    for i in range(pca_factors+1):
        pnlcs.extend([np.cumsum(sum(pnl[i]))])

    plot_area_pnl(pnlcs, save=True, legend=True)

    # graph_days= 14; offset = 0
    # x = x[offset:graph_days]
    # for i in range(pca_factors):
    #     plt.bar(x,sum(pnl[1+i])[offset:graph_days])
    # axis2 = plt.twiny()
    # axis2.set_xticks([])
    # axis2.plot(x, sum(pnl[0])[offset:graph_days], color='black')
    # plt.show()
    #
    # for i in range(pca_factors):
    #     plt.bar('Total',pnlcs[i+1][-1])
    # axis2 = plt.twiny()
    # axis2.set_xticks([])
    # axis2.plot('Total',pnlcs[0][-1])
    # plt.show()


    print("done")
