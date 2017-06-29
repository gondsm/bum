#!/usr/bin/python
# Futures
from __future__ import division
from __future__ import print_function

# Standard Lib
import yaml
import matplotlib.pyplot as plt
import itertools
import numpy as np

# ROS
import rospy
from bum_ros.msg import Likelihood, Tuple, Evidence

# Custom
import radar_chart_example as radar_chart

GCD = None

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


def reset_population(population, gcd):
    """ Resets user population back to a uniform distribution. 

    This is done according to the GCD and the population generated is of the
    form population["characteristic_id"] = {[evidence, identity]: characteristic_value}
    meaning that there is a main entry for each characteristic, and within it
    the values are contained in a dict for every combination of evidence and
    identity (indexed as a single tuple).
    """
    # Population is split by characteristics
    for char in gcd["C"]:
        # Create new dictionary for this characteristic
        population[char] = dict()

        # Create evidence structure for this variable
        evidence_structure = []
        for key in gcd["C"][char]["input"]:
            evidence_structure.append(gcd["E"][key]["nclasses"])

        # Initialize characteristics for all evidence combination
        a = [range(0, elem) for elem in evidence_structure]
        for i in range(gcd["nusers"]):
            iterator = itertools.product(*a)
            for comb in iterator:
                population[char][comb + (i,)] = np.random.randint(0, gcd["C"][char]["nclasses"])


def calc_estimation_error(population, gt_data, gcd):
    """ Given a population, a gcd and the ground truth, this function
    calculates the total estimation error.

    For now, we assume only characteristics that are fixed to the user,
    independent of evidence.

    TODO: Test if dict can be called by evidence. If so, compare for all
    combinations. If not, assume that characteristic is fixed to the user.
    """
    # Initialize null error
    error = 0

    # Calculate error for each characteristic
    for char in gcd["C"]:
        # For each "item" in the population, i.e. each combination of evidence
        # and identity.
        for key, item in population[char].iteritems():
            try:
                # Calculate error by simple sum
                error += abs(gt_data[char][key[-1]] - item)
            except KeyError:
                # We may not have ground truth for some users, and that is okay.
                # TODO: Make this more robust
                pass

    # Return our total error
    return error


def iterative_evaluation(exec_log_filename, gt_log_filename, gcd_filename):
    """ Analyses the exec_log and gt_log to produce figures of the execution in
    time, like the ones we have for simulations. 
    """
    # Load external data
    with open(exec_log_filename) as exec_log:
        exec_data = yaml.load(exec_log)
    with open(gt_log_filename) as gt_log:
        gt_data = yaml.load(gt_log)
    with open(gcd_filename) as gcd_file:
        gcd = yaml.load(gcd_file)

    # Empty population
    population = dict()
    reset_population(population, gcd)

    # Initialize error
    error = []

    # Iterate for all data points
    for it_data in exec_data:
        # W
        error.append(calc_estimation_error(population, gt_data, gcd))


if __name__=="__main__":
    # Initialize ROS node
    rospy.init_node('evaluator_node')
    rospy.loginfo("BUM Evaluator ROS node started!")

    # Initialize subscribers
    rospy.Subscriber("bum/tuple", Tuple, tuple_callback)
    #rospy.Subscriber("bum/evidence", Evidence, evidence_callback)

    gcd_filename = ""
    gt_log_file = ""
    exec_log_file = ""

    # Get parameters
    try:
        gcd_filename = rospy.get_param("bum_ros/gcd_file")
    except KeyError:
        rospy.logfatal("Could not get GCD file name parameter")
        exit()
    try:
        gt_log_file = rospy.get_param("bum_ros/gt_log_file")
    except KeyError:
        rospy.logfatal("Could not get gt log file name parameter")
        exit()
    try:
        exec_log_file = rospy.get_param("bum_ros/exec_log_file")
    except KeyError:
        rospy.logfatal("Could not get exec log file name parameter")
        exit()

    # Read GCD file
    with open(gcd_filename, "r") as gcd_file:
        GCD = yaml.load(gcd_file)

    for key in GCD["C"]:
        latest_char[key] = 0

    iterative_evaluation(exec_log_file, gt_log_file, gcd_filename)

    exit()

    # Let her spin!
    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        plot_characteristics(latest_char)
        r.sleep()