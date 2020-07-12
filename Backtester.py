from engine import *
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


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
        self.long_pnl = None

    def cache_rand(mySurf,persist=True):
        '''
        Cache normal random variables for simulation in pickle object
        :param mySurf: A volSurf object with histvols value initialized
        :param persist: If True, save results in a pickle
        :return: Data that was cached
        '''
        rands = np.random.normal(0, 1, mySurf.histvols.shape)
        if persist:
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

    def get_rands(self, refresh, persist=True):
        '''
        Gets random variables used in the simulation
        :param refresh: Bool. Refresh cache or not
        :param persist: If true, save file in a pickle
        :return: None. Sets instance variable
        '''
        if os.path.exists(Backtester.rand_fn) and not refresh:
            self.myrands = pickle.load( open(Backtester.rand_fn, "rb" ) )
        else:
            self.myrands = Backtester.cache_rand(mySurf, persist=persist)
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

    def init_pos(self, invest=1e6, random=True):
        '''
        Initialize backtesting positions
        :param invest: Initial amount invested (float)
        :param random: If false, positions remain constant throughout; otherwise, random each day
        :return: None. Sets instance variable
        '''
        self.pos = self.myrands.copy()
        self.limit = invest*5 #invest/5 #invest #limit for individual contracts
        if not random:
            self.pos[:,0] = 1
            self.pos[:, 1:] = 0
        self.pos[:, 0] = self.pos[:, 0]* myBT.mycontw
        self.pos[:, 0] = invest * self.pos[:, 0] / sum(self.pos[:, 0])
        self.pos[:, 0][self.pos[:, 0] > self.limit] = self.limit  # Limit size of long positions
        self.pos[:, 0][self.pos[:, 0] < -self.limit] = -self.limit  # Limit size of shorts
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
        for day in range(1,sim_days):
            self.pos[:, day] = self.pos[:,day-1]*(1 + self.mycontw*self.pos[:,day])
            self.pos[:, day][self.pos[:, day] > self.limit] = self.limit #Limit size of long positions
            self.pos[:, day][self.pos[:, day] < -self.limit] = -self.limit #Limit size of shorts
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

    def sim_pnl_reg(self, batch, fact_index=None):
        '''
        Simulate positions several times to pnl with differing positions
        :param batch: (int) Number of times to simulate positions
        :param fact_index: (list of int) Specify which factors we want to measure
        :return: None changes long pnl instance variable
        '''
        num_days = np.shape(self.mySurf.histvols)[1]-1
        num_factors = self.mySurf.pca_num_factors() if fact_index is None else len(fact_index)
        factor_hist = self.mySurf.proj_fact(diff=True)
        self.long_pnl = np.zeros((num_factors+1,batch*num_days)) #initialize pnl vectors
        fact_index = [x for x in range(num_factors)] if fact_index is None else fact_index
        for day in range(batch):
            self.init_pos()
            self.run_bt()
            self.long_pnl[0,day*num_days:(day+1)*num_days] = np.sum(myBT.get_pnl(factor=False),axis=0)
            for fact in range(len(fact_index)):
                self.long_pnl[fact+1,day*num_days:(day+1)*num_days] = \
                    np.sum(myBT.get_pnl(factor=True, shifts=factor_hist[fact_index[fact]], pc=self.mySurf.pca.components_[fact_index[fact]]),axis=0)
            self.get_rands(refresh=True,persist=False)
        return None

    def reg_long_sim(self,qq=False):
        '''
        Display regression results for comparing simulated pnl to factor pnl
        :param qq: If true, print a qq plot
        :return: None. Results will be printed
        '''
        model = sm.OLS(self.long_pnl[0, :].transpose(), self.long_pnl[1:, :].transpose()).fit()
        print(model.summary())
        if qq:
            res = model.resid
            fig = sm.qqplot(res)
            plt.show()
        return None

    def get_cumsum(self, pnl, num_factors):
        '''
        :param pnl: list of list containing pnl. First list is total pnl. Remaining are factors
        :param num_factors: Number of factors to consider
        :return: list of list with cummulative sums
        '''
        pnlcs = []
        for i in range(num_factors + 1):
            pnlcs.extend([np.cumsum(sum(pnl[i]))])
        return pnlcs

    def get_expl_factors(self, pnlcs, num_factors):
        '''
        Returns an index which indicates the top explainitory pca factors for the simulation
        :param pnlcs: list of list cummulative pnl with first list the total pnl
        :return: (list) Index of most impacting factors
        '''
        unexplained = [np.mean(1-f/pnlcs[0]) for f in pnlcs[1:]]
        return sorted(range(1,len(pnlcs)), key= lambda i : unexplained[i-1])[:num_factors]

if __name__ == "__main__":
    investment = 1e6; random = False; pca_factors = 5; fact_opt = False

    #Initialize vol surface
    mySurf = VolSurf('data_download_semi_clean.csv',h1=0.1,h2=0.1)

    # Run backtest
    myBT = Backtester(mySurf,refresh=False)
    myBT.mySurf.get_pca(pca_factors, cov=True)
    # myBT.sim_pnl_reg(batch=1,fact_index=[0,1])
    # myBT.reg_long_sim(qq=False)
    fact_impact = myBT.mySurf.proj_fact(diff=True)
    myBT.init_pos(invest=investment, random=random)
    myBT.run_bt()
    # pickle.dump(myBT, open('myBT.p', "wb"))
    # myBT = pickle.load(open( 'myBT.p', "rb" ))

    #Compute PNL
    pnl = []
    pnl.extend([myBT.get_pnl(factor=False)])
    for i in range(pca_factors):
        pnl.extend([myBT.get_pnl(factor=True,shifts=fact_impact[i], pc=myBT.mySurf.pca.components_[i])])



    #Compute cumsum for each pnl
    pnlcs = myBT.get_cumsum(pnl, pca_factors)
    best_fact = myBT.get_expl_factors(pnlcs, 5) if fact_opt else [x+1 for x in range(len(pnlcs)-1)]
    plot_stacked_bar_pnl(pnlcs=pnlcs,save=True,legend=True,filter=best_fact,fact_line=True)



    print("done")
