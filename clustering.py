#python3
# Based on http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py
# https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
import itertools

import numpy as np
import numpy.random as rnd
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D


from sklearn import mixture

import plot_user

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_clusters(means, covariances):
    """ Given the result of the E-M algorithm, this function plots the clusters
    that were determined.
    """
    colors = [[1, 0, 0, 0.2],
              [0, 1, 0, 0.2],
              [0, 0, 1, 0.2]]
    color_iter = itertools.cycle(colors)

    def plot_ellipsoid(radii, center, axes):
        """ Plots an ellipsoid in the given axes
        radii = [x,y,z]
        center = [x,y,z]
        """
        #coefs = (1, 1, 2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
        # Radii corresponding to the coefficients:
        #rx, ry, rz = 1/np.sqrt(coefs)
        rx, ry, rz = radii

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + center[2]

        # Plot:
        axes.plot_surface(x, y, z,  rstride=4, cstride=4, linewidth=0, color=next(color_iter))

        ax.set_xlabel("C_1")
        ax.set_ylabel("C_2")
        ax.set_zlabel("C_3")

    # Create figure
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
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

    # Set plot limits
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])

    # Show plot
    plt.show()


def original():
    def plot_results(X, Y_, means, covariances, index, title):
        splot = plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        plt.xlim(-9., 5.)
        plt.ylim(-3., 6.)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)


    # Number of samples per component
    n_samples = 500

    # Generate random sample, two components
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
                 'Gaussian Mixture')

    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                            covariance_type='full').fit(X)
    plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
                 'Bayesian Gaussian Mixture with a Dirichlet process prior')

    plt.show()

def test():
    # Generate a couple of really clear clusters
    X = []
    for i in range(50):
        X.append([np.random.randint(4), np.random.randint(4), np.random.randint(4)])
    for i in range(50):
        X.append([np.random.randint(5,10), np.random.randint(5,10), np.random.randint(5,10)])

    # Show the users
    plot_user.plot_users(X)

    # Cluster
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X)
    #print(gmm.means_, gmm.covariances_)

    # Plot clusters
    plot_clusters(gmm.means_, gmm.covariances_)

if __name__=="__main__":
    #original()
    test()