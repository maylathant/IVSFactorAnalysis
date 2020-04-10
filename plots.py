import matplotlib.pyplot as plt
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
    plt.bar([str(x) for x in range(1,len(importance)+1)],height=importance)
    plt.savefig("factorimportance.pdf")
    plt.show()
    return None

def plt_proj_time(mySurf, pca_factors):
    '''
    Plot a time series of the projected factors
    :param mySurf: Implied volatility object
    :param pca_factors: Number of pca factors
    :return: None
    '''
    proj = mySurf.proj_fact()
    labels = ['Factor {}'.format(l+1) for l in range(np.shape(proj)[0])]
    for i in range(pca_factors):
        plt.plot(proj[i],label=labels[i])
    plt.xlabel('Days')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
    plt.show()
    return None