import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plt_surf(mySurf, graph_scn,diff=False,save=False,title=None):
    '''
    Surface plot for an implied volatility surface for a particular scenario or date
    :param mySurf: VolSurf object
    :param graph_scn: Date or scenario to plot
    :param diff: If true, use vol % differences instead of volatility quote
    :param save: If true, save a file of the image
    :return: None
    '''
    tim = np.linspace(0.08,4,30)
    money = np.linspace(0.4,1.6,30)
    tim, money = np.meshgrid(tim,money)
    vecfunc = np.vectorize(mySurf.nwInterp)
    ivs = vecfunc(money,tim,graph_scn,diff)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_zlim3d(0, 0.10) #If scale is very small
    ax.plot_surface(money,tim,ivs)
    ax.view_init(30,60)
    if title is not None:
        plt.title(title)
    if save:
        plt.savefig(str(graph_scn) + '.pdf')
    plt.show()
    return None

def plt_importance(mySurf):
    '''
    Plot factor importance in a bar chart
    :param mySurf: VolSurf object
    :return: None
    '''
    plt.title('PCA Factor Importance')
    plt.xlabel('PCA Factor')
    importance = mySurf.pca.singular_values_/sum(mySurf.pca.singular_values_)
    plt.bar([x for x in range(1,len(importance)+1)],height=importance)
    plt.savefig("factorimportance.pdf")
    plt.show()
    return None

def plt_proj_time(mySurf, pca_factors,save=False,title=None):
    '''
    Plot a time series of the projected factors
    :param mySurf: Implied volatility object
    :param pca_factors: Number of pca factors
    :param save: If true, save a file of the image
    :param title: The title for the image
    :return: None
    '''
    proj = mySurf.proj_fact()
    labels = ['Factor {}'.format(l+1) for l in range(np.shape(proj)[0])]
    for i in range(pca_factors):
        plt.plot(proj[i],label=labels[i])
    plt.xlabel('Days')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
    if save:
        plt.savefig(title)
    plt.show()
    return None

def plt_surf_mult(mySurf, graph_scn,diff=False,save=False,title=None,color=None):
    '''
    Plot multiple surfaces in the same figure
    :param mySurf: VolSurf object
    :param graph_scn: (list) Dates or scenarios to plot
    :param diff: If true, use vol % differences instead of volatility quote
    :param save: If true, save a file of the image
    :param title: List of titles for the images
    :param color: List of colors for the plots
    :return: None
    '''
    tim = np.linspace(0.08,4,30)
    money = np.linspace(0.4,1.6,30)
    tim, money = np.meshgrid(tim,money)
    vecfunc = np.vectorize(mySurf.nwInterp)
    figlen = int(len(graph_scn)/2)
    fig = plt.figure()
    for i in range(figlen):
        for j in range(2):
            ivs = vecfunc(money, tim, graph_scn[i*2+j], diff)
            ax = fig.add_subplot(figlen,2,i*2+j+1,projection='3d')
            if color is None:
                ax.plot_surface(money,tim,ivs)
            else:
                ax.plot_surface(money, tim, ivs,color=color[i*2+j])
            ax.view_init(30,60)
            ax.set_xlabel('Moneyness')
            ax.set_ylabel('Years to Expiry')
            ax.set_zlabel('Volatility Shift')
            if title is not None:
                ax.set_title(title[i*2+j])
    if save:
        plt.savefig(str(''.join(graph_scn)) + '.pdf')
    plt.show()
    return None

def plot_area_pnl(pnlcs, save=False, title=None, legend=False, filter=None):
    '''
    Plot area chart of cumulative PNL
    :param pnlcs: A list of cumulative pnls saved in numpy. First element should
    be the baseline pnl series
    :param filter: List representing the index to keep in pnlcs for attribution
    :return: None. Figure saved locally
    '''
    if filter is not None:
        to_plot = [pnlcs[i] for i in filter]
        labels = ['Factor ' + str(x) for x in filter]
    else:
        to_plot = pnlcs[1:]
        labels = ['Factor ' + str(x) for x in range(1, len(pnlcs) + 1)]
    x = [x for x in range(len(pnlcs[0]))]
    plt.stackplot(x, to_plot, labels=labels,colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728','grey'])
    plt.ylabel("PNL")
    plt.xlabel("Trading Day")
    if legend:
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=4, mode="expand", borderaxespad=0.)
    axis2 = plt.twiny()
    axis2.set_xticks([])
    axis2.plot(x, pnlcs[0], color='black', label='Total PNL')
    if title is not None:
        plt.title(title)
    if legend:
        plt.legend()
    if save:
        plt.savefig('pnl_explain.png')
    plt.show()
