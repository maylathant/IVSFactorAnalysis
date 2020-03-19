import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plt_surf(mySurf, graph_scn):
    '''
    Surface plot for an implied volatility surface for a particular scenario or date
    :param mySurf: VolSurf object
    :param graph_scn: Date or scenario to plot
    :return: None
    '''
    tim = np.linspace(0.08,4,30)
    money = np.linspace(0.4,1.6,30)
    tim, money = np.meshgrid(tim,money)
    vecfunc = np.vectorize(mySurf.nwInterp)
    ivs = vecfunc(money,tim,graph_scn)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(money,tim,ivs)
    ax.view_init(30,60)
    plt.savefig(graph_scn + '.pdf')
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