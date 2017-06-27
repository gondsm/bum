#!/usr/bin/python
# Futures
from __future__ import division
from __future__ import print_function

# Standard Lib
import yaml
import matplotlib.pyplot as plt

# ROS
import rospy
from bum_ros.msg import Likelihood, Tuple, Evidence

# Custom
import radar_chart_example as radar_chart

GCD = None

# TODO: Receive from input
gcd_filename = "/home/vsantos/catkin_ws/src/bum_ros/bum_ros/config/data_gathering.gcd"
gt_log_file = "/home/vsantos/Desktop/bum_ros_data/gt_log.yaml"

# Global object for maintaining the figure alive
radar_fig = None

# The latest set of characteristics, for plotting
latest_char = dict()

def plot_characteristics(characteristics):
    """ This function plots the latest data received during runtime.

    characteristics should be a char_id -> value dict """
    # Get character names from input
    char_names = []
    char_vals = []
    for key in characteristics:
        char_names.append(key)
        char_vals.append(characteristics[key])

    # Create figure if it doesn't exist
    global radar_fig
    if not radar_fig:
        radar_fig = plt.figure(figsize=(9, 9), tight_layout = True)

    # Draw radar chard
    N = len(char_names)
    theta = radar_chart.radar_factory(N, frame='circle')
    colors = ['b', 'r', 'g', 'm', 'y']
    ax = radar_fig.add_subplot(1,1,1, projection='radar')
    color_index = 0
    ax.plot(theta, char_vals, color=colors[color_index])
    ax.fill(theta, char_vals, facecolor=colors[color_index], alpha=0.25)
    ax.set_varlabels(char_names)
    ax.set_ylim([0, 10])

    # Show plot
    plt.show(block=False)
    radar_fig.canvas.draw()


def tuple_callback(msg):
    """ Responsible for processing new tuples, adds them to the latest characteristics. """
    latest_char[msg.char_id] = msg.characteristic


def plot_single_user_run():
    """ Plots a full run of the system with a single user. """
    # Bogus data for now
    error_vec = [10, 10, 10, 8, 8, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2]
    c_vec = [
             [10, 10, 10, 10, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
             [6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8],
             [2, 2, 2, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3]
            ]

    # Create figure
    fig = plt.figure()

    # Plot error
    plt.subplot(2,1,1)
    plt.plot(error_vec)
    plt.ylim([0, 1.1*max(error_vec)])
    #plt.xlim([0, len(error_vec)])
    plt.ylabel("Estimation Error")

    # Plot characteristics
    plt.subplot(2,1,2)
    plt.hold(True)
    for i, vec in enumerate(c_vec):
        plt.plot(vec, label="C{}".format(i+1))
    plt.ylim([0, 11])
    plt.legend()
    plt.ylabel("Characteristics")
    plt.xlabel("Iterations (k)")

    # Show/save plot
    plt.savefig("single_user_run.pdf")


def plot_multi_user_run():
    """ Plots a full run with multiple users. """
    # Bogus data for now
    #error_vec = [89, 89, 89, 89, 81, 81, 81, 81, 81, 75, 75, 75, 75, 75, 75, 75, 62, 62, 62, 62, 62, 62, 62, 62, 50, 50, 50, 50, 43, 43, 43, 43, 43, 43, 34, 34, 34, 34, 34, 20, 20, 20, 20, 20, 20, 12, 12, 11, 12, 11, 10, 11, 12, 11, 10, 9, 7, 11, 9, 10, 9]
    error_vec = [89, 89, 89, 89, 81, 81, 81, 81, 81, 75, 75, 75, 75, 75, 75, 75, 62, 62, 62, 62, 62, 62, 62, 62, 50, 50, 50, 50, 43, 43, 43, 43, 43, 43, 34, 34, 34, 34, 34, 20, 20, 20, 20, 20, 20, 20, 21, 19, 20, 18, 22, 21, 19, 22, 20, 20, 21, 19, 21, 20, 20]
    users_vec = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

    # Plot error
    plt.subplot(2,1,1)
    plt.plot(error_vec)
    plt.ylim([0, 1.1*max(error_vec)])
    #plt.xlim([0, len(error_vec)])
    plt.ylabel("Estimation Error")


    # Plot number of users
    plt.subplot(2,1,2)
    plt.plot(users_vec)
    plt.ylim([0, 1.1*max(users_vec)])
    plt.ylabel("Number of Users")
    plt.xlabel("Iterations (K)")

    # Show/save plot
    #plt.show()
    plt.savefig("multi_user_run.pdf")


if __name__=="__main__":
    # Initialize ROS node
    rospy.init_node('evaluator_node')
    rospy.loginfo("BUM Evaluator ROS node started!")

    # Initialize subscribers
    rospy.Subscriber("bum/tuple", Tuple, tuple_callback)
    #rospy.Subscriber("bum/evidence", Evidence, evidence_callback)
    
    # Get parameters
    try:
        gcd_filename = rospy.get_param("bum_ros/gcd_file")
    except KeyError:
        rospy.logwarn("Could not get GCD file name parameter")
    try:
        gt_log_file = rospy.get_param("bum_ros/gt_log_file")
    except KeyError:
        rospy.logwarn("Could not get gt log file name parameter")

    # Read GCD file
    with open(gcd_filename, "r") as gcd_file:
        GCD = yaml.load(gcd_file)

    for key in GCD["C"]:
        latest_char[key] = 0

    #plot_single_user_run()
    #plot_multi_user_run()
    #exit()

    with open(gt_log_file, "r") as gt_log:
        gt = yaml.load(gt_log)    
    print(gt)
    exit()

    # Let her spin!
    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        plot_characteristics(latest_char)
        r.sleep()