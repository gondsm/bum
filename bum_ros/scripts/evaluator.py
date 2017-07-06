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

# Scikit-learn
from sklearn import mixture

# Custom
import radar_chart_example as radar_chart
import bum_utils as utils

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


def plot_single_user_run(error_vec, c_dict):
    """ Plots a full run of the system with a single user.
    error_vec is the timeseries of errors calculated, c_vec is a dict
    of timeseries, one for each characteristic, indexed by characteristic
    name.
    """
    # Create figure
    fig = plt.figure()

    # Plot error
    plt.subplot(2,1,1)
    plt.plot(error_vec)
    plt.ylim([0, 1.1*max(error_vec)])
    plt.xlim([0, len(error_vec)-1])
    plt.ylabel("Estimation Error")

    # Plot characteristics
    plt.subplot(2,1,2)
    plt.hold(True)
    for key in sorted(c_dict):
        plt.plot(c_dict[key], label=key)
    plt.ylim([0, 5])
    plt.xlim([0, len(error_vec)-1])
    plt.legend()
    plt.ylabel("Characteristics")
    plt.xlabel("Iterations (k)")

    # Show/save plot
    #plt.show()
    plt.savefig("single_user_run.pdf")


def plot_multi_user_run(error_vec, users_vec):
    """ Plots a full run with multiple users. """
    # Plot error
    plt.subplot(2,1,1)
    plt.plot(error_vec)
    plt.xlim([0, len(error_vec)-1])
    plt.ylim([0, 1.1*max(error_vec)])
    plt.ylabel("Estimation Error")

    # Plot number of users
    plt.subplot(2,1,2)
    plt.plot(users_vec)
    plt.xlim([0, len(users_vec)-1])
    plt.ylim([0, 1.1*max(users_vec)])
    plt.ylabel("Number of Users")
    plt.xlabel("Iterations (K)")

    # Show/save plot
    plt.show()
    #plt.savefig("multi_user_run.pdf")


def calc_estimation_error(population, gt_data, gcd, user_id=None):
    """ Given a population, a gcd and the ground truth, this function
    calculates the total estimation error.

    For now, we assume only characteristics that are fixed to the user,
    independent of evidence.

    If user_id is received, the error is calculated for that single user.

    TODO: Test if dict can be called by evidence. If so, compare for all
    combinations. If not, assume that characteristic is fixed to the user.
    """
    # Initialize null error
    error = 0

    # Calculate error for each active characteristic
    for char in gcd["Config"]["Active"]:
        # For each "item" in the population, i.e. each combination of evidence
        # and identity.
        for key, item in population[char].iteritems():
            # If we're calculating for a single ID, we ignore the rest
            if user_id is not None and key[-1] != user_id:
                continue
            try:
                # Calculate error by simple sum
                error += abs(gt_data[char][key[-1]] - item)
            except KeyError:
                # We may not have ground truth for some users, and that is okay.
                # TODO: Make this more robust
                pass

    # Return our total error
    return error


def iterative_evaluation(exec_log_filename, gt_log_filename, gcd_filename, user_id=None):
    """ Analyses the exec_log and gt_log to produce figures of the execution in
    time, like the ones we have for simulations. 

    If user_id is set as an integer, this function will be run for a single
    user only, producing a different plot.
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
    utils.reset_population(population, gcd)

    # Initialize error
    error = []
    # Initialize a set of users
    known_users = []
    # Initialize the user counter (timeseries)
    n_users = []

    if user_id is None:
        # Iterate for all data points
        for it_data in exec_data:
            # Integrate new classification(s) into population
            for char in it_data["C"]:
                population[char][tuple(it_data["Evidence"]) + (it_data["Identity"],)] = it_data["C"][char]
            if it_data["Identity"] not in known_users:
                known_users.append(it_data["Identity"])
            n_users.append(len(known_users))
            # Update error
            error.append(calc_estimation_error(population, gt_data, gcd))
        # And plot
        plot_multi_user_run(error, n_users)
    else:
        # Add our single user to the known users
        known_users.append(user_id)
        # Initialize characteristic history
        char_history = dict()
        # Iterate for all data points
        for it_data in exec_data:
            # Ignore data points not for our single user
            if it_data["Identity"] != user_id:
                continue
            # Integrate new classification(s) into population
            for char in it_data["C"]:
                population[char][tuple(it_data["Evidence"]) + (it_data["Identity"],)] = it_data["C"][char]
            # Get current characteristics from population
            for char in ["C1", "C2"]:
                try:
                    char_history[char].append(population[char][(user_id,)])
                except KeyError:
                    char_history[char] = [population[char][(user_id,)]]
            n_users.append(len(known_users))
            # Update error
            error.append(calc_estimation_error(population, gt_data, gcd, user_id))
        # And plot
        plot_single_user_run(error, char_history)


def fault_tolerance_evaluation(exec_log_filename, gt_log_filename, gcd_filename):
    """ An iterative evaluation of the system's fault-tolerance, using a
    clustering technique to get missing user characteristics.
    """
    # Load external data
    with open(exec_log_filename) as exec_log:
        exec_data = yaml.load(exec_log)
    with open(gt_log_filename) as gt_log:
        gt_data = yaml.load(gt_log)
    with open(gcd_filename) as gcd_file:
        gcd = yaml.load(gcd_file)

    # Define two user populations
    # Population 1 will be used for training, and population 2 will be used
    # to test the system's fault-tolerance by determining all characteristics
    # from the clusters obtained from the training population.
    # TODO: Make this selection automatic
    pop_1 = [1, 2, 5, 6, 7]
    pop_2 = [3, 4]

    # Empty population
    population = dict()
    utils.reset_population(population, gcd, pop_1)

    # "Train" population from the exec_data we have
    for it_data in exec_data:
        if it_data["Identity"] not in pop_1:
            continue
        # Integrate new classification(s) into population
        for char in it_data["C"]:
            population[char][tuple(it_data["Evidence"]) + (it_data["Identity"],)] = it_data["C"][char]

    # Cluster training population
    means, cov = utils.cluster_population(population, gcd, num_clusters=1)

    # Estimate missing characteristics


    # Calculate error
    calc_estimation_error(population, gt_data, gcd)


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

    # Generate offline plots:
    #iterative_evaluation(exec_log_file, gt_log_file, gcd_filename)
    #iterative_evaluation(exec_log_file, gt_log_file, gcd_filename, user_id=1)
    fault_tolerance_evaluation(exec_log_file, gt_log_file, gcd_filename)

    exit()

    # Let her spin!
    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        plot_characteristics(latest_char)
        r.sleep()