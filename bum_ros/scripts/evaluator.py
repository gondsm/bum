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
gcd_filename = "/home/vsantos/catkin_ws/src/bum_ros/bum_ros/config/caracteristics.gcd"

# Global object for maintaining the figure alive
radar_fig = None

# The latest set of characteristics, for plotting
latest_char = dict()

def plot_characteristics(characteristics):
    """ This function plots the latest data received.

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


if __name__=="__main__":
    # Initialize ROS node
    rospy.init_node('evaluator_node')
    rospy.loginfo("BUM Data Manager ROS node started!")

    # Initialize subscribers
    rospy.Subscriber("bum/tuple", Tuple, tuple_callback)
    #rospy.Subscriber("bum/evidence", Evidence, evidence_callback)
    
    # Read GCD file
    with open(gcd_filename, "r") as gcd_file:
        GCD = yaml.load(gcd_file)

    for key in GCD["C"]:
        latest_char[key] = 0

    # Let her spin!
    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        plot_characteristics(latest_char)
        r.sleep()