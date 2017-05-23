import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
#import colormaps as cmaps


def plot_gaussian():

    #Parameters to set
    mu_x = 0
    sigma_x = np.sqrt(3)

    mu_y = 0
    sigma_y = np.sqrt(15)

    #Create grid and multivariate normal
    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X, Y = np.meshgrid(x,y)
    Z = bivariate_normal(X,Y,sigma_x,sigma_y,mu_x,mu_y)

    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z,cmap="jet",linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=15., azim=45)
    plt.show()

    #plt.savefig("gaussian.png")

def plot_evidence_signal():
    fig = plt.figure(figsize=[6,4])
    fig.patch.set_alpha(0.0)
    x = range(1,10)
    markerline, stemlines, baseline = plt.stem(x, np.random.randint(2, 10, size=[len(x)]), '-.')
    #plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    ax = plt.gca()
    ax.patch.set_facecolor('red')

    plt.ylabel("$\mathbf{E}_i$", fontsize=40)
    plt.xlabel("$i$", fontsize=40)
    plt.tight_layout()

    plt.savefig("evidence.png", transparent=True)


if __name__=="__main__":
    # Configure Matplotlib
    mpl.rcParams['ps.useafm'] = True
    #mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True
    #mpl.rcParams['text.latex.unicode']=False
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['axes.ymargin'] = 0.1
    #plot_evidence_signal()
    plot_gaussian()