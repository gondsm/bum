""" This simple script serves as a test bench for plotting users using
matplotlib scatter plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def plot_users(users):
    # Set mpl parameters
    mpl.rcParams['legend.fontsize'] = 10
    #mpl.rcParams['text.usetex'] = True

    colors=["blue", "red", "orange"]

    # Initialize  and adjust figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_zlim([0,10])
    ax.set_xlabel("C_1")
    ax.set_ylabel("C_2")
    ax.set_zlabel("C_3")
    ax.view_init(elev=15., azim=45)

    # Plot/Save
    plt.hold(True)
    for i, user in enumerate(users):
        # Point
        #print(users[i])
        ax.plot([user[0]], [user[1]], [user[2]], 'o', label='User {}'.format(i+1))#, color=colors[i])
        # Lines
        #ax.plot([x[i], x[i]], [y[i], y[i]], [0, z[i]], color=colors[i])
        #ax.plot([x[i], x[i]], [0, y[i]], [z[i], z[i]], color=colors[i])
        #ax.plot([0, x[i]], [y[i], y[i]], [z[i], z[i]], color=colors[i])
        # Crosses
        #ax.plot([x[i]], [y[i]], [0], 'x', color=colors[i])
        #ax.plot([x[i]], [0], [z[i]], 'x', color=colors[i])
        #ax.plot([0], [y[i]], [z[i]], 'x', color=colors[i])
        
    #ax.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
    #ax.legend(numpoints=1, ncol=3)
    plt.show()
    #plt.savefig("users_example.pdf")



if __name__ == "__main__":
    plot_users([[3, 5, 6], [2, 5, 9], [1, 5, 7]])