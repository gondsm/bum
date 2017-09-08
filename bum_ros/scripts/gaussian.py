""" The gaussian script
Used for plotting example figures for illustrations.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
import itertools
from scipy import linalg
#import colormaps as cmaps

def plot_matching_fig(filename=None, title=None):
    """ Given the result of the E-M algorithm, this function plots the clusters
    that were determined. If a filename is given, the plot is saved on that
    file.
    """
    colors = [[1, 0, 0, 0.2],
              [0, 1, 0, 0.2],
              [0, 0, 1, 0.2],
              [1, 0, 1, 0.2]]
    color_iter = itertools.cycle(colors)

    means = [[7,2,2], [2,7,7]]
    covariances = np.array(
                    [[[  1.5,  -1.0,   1.5],
                    [ -1.0,   1.2,  -1.2],
                    [  1.0,  -1.2,   1.5]],
                    [[  1.5,  -1.0,   1.5],
                    [ -1.0,   1.2,  -1.2],
                    [  1.0,  -1.2,   1.5]]]
                )
    covariances[0] = covariances[0]/3
    covariances[1] = covariances[1]/2.5

    def plot_ellipsoid(radii, center, axes):
        """ Plots an ellipsoid in the given axes
        radii = [x,y,z]
        center = [x,y,z]
        """
        # Unpack radii
        rx, ry, rz = radii

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + center[2]
        #z = [[0]*len(z)]

        # Plot:
        axes.plot_surface(x, y, z,  rstride=4, cstride=4, linewidth=0, color=next(color_iter))

    # Create figure
    fig = plt.figure()  # Square figure
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=45)

    # Plot clusters
    for mean, cov in zip(means, covariances):
        # Determine axis radii
        v, w = linalg.eigh(cov)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # Plot
        plot_ellipsoid(v, mean, ax)

    # Plot user and lines
    user = [9,4,2]
    ax.plot([user[0]], [user[1]], [user[2]], 'o')
    #ax.plot([7, user[0]], [2, user[1]], [0, user[2]])
    #ax.plot([2, user[0]], [7, user[1]], [0, user[2]])

    # Set plot limits and labels
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    ax.set_xlabel(r"$C_1$")
    ax.set_ylabel(r"$C_2$")
    ax.set_zlabel(r"$C_3$")
    if title is not None:
        plt.title(title)

    # Show/save plot
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()


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
    #plot_gaussian()

    plot_matching_fig("user_matched.pdf")