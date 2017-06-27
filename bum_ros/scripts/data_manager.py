#!/usr/bin/python
""" The data_manager scripts
This node is responsible for managing all of the lig files needed by the system.
It can play back an evidence log for re-testing certain data, and it saves
evidence (containing both hard examples and non-classified evidence) and
execution (containing all of the system's responses) logs, for later analysis
and playback.
"""
# Futures
from __future__ import division
from __future__ import print_function

# Standard Lib
import yaml
import os
import random

# ROS
import rospy
from bum_ros.msg import Likelihood, Tuple, Evidence

GCD = None

# Global variables with defaults, also read from parameters
log_hard = True

def log_evidence(evidence, identity, characteristic, char_id, filename):
    """ This function adds a new record to the evidence log. 

    Charateristics are received as a value and an string id. 
    Evidence is received as a dict ev_id -> value
    """
    # Opens the file, or creates it if it doesn't exist
    try:
        data_file = open(filename, 'a')
    except IOError:
        data_file = open(filename, 'w')

    # Prepare data dictionary
    data = []
    data_dict = dict()
    data_dict["Evidence"] = evidence
    data_dict["Identity"] = identity
    if characteristic is not None:
        data_dict["C"] = dict()
        data_dict["C"][char_id] = characteristic
    data.append(data_dict)

    # Write to file
    rospy.loginfo("Logging evidence: {}.".format(data))
    data_file.write(yaml.dump(data, default_flow_style=False))

    # ... and close the file
    data_file.close()


def log_classification(evidence, identity, characteristic, char_id, entropy, filename):
    """ This function adds a new record to the execution log, for later
    evaluation.
    """
    # Opens the file, or creates it if it doesn't exist
    try:
        data_file = open(filename, 'a')
    except IOError:
        data_file = open(filename, 'w')

    # Prepare data dictionary
    data = []
    data_dict = dict()
    data_dict["Evidence"] = evidence
    data_dict["Identity"] = identity
    data_dict["Entropy"] = entropy
    data_dict["C"] = dict()
    data_dict["C"][char_id] = characteristic
    data.append(data_dict)

    # Write to file
    rospy.loginfo("Logging classification: {}.".format(data))
    data_file.write(yaml.dump(data, default_flow_style=False))

    # ... and close the file
    data_file.close()


def playback_evidence(filename):
    """ This funcion plays back evidence recorded in filename. 

    For each entry in the log, if it has a characteristic associated, it is
    interpreted as hard evidence and published as a tuple message. If it does
    not, it is published as simple Evidence message.

    Messages are published 0.5 secs apart.

    If a directory is passed instead of a file name, the function reads all
    files in the directory and interleaves their playback.
    """
    rospy.loginfo("Playing back evidence")

    # Read file or directory
    if os.path.isdir(filename):
        # Inform
        rospy.loginfo("Opening directory for playback.")
        # Get a list of tiles
        files = []
        for (dirpath, dirnames, filenames) in os.walk(filename):
            # Get files avoiding any ground truth that might be there
            files.extend([f for f in filenames if "gt_log" not in f])
        # Prepend the full path so we can open them later
        files = [os.path.join(filename, f) for f in files]
        # Read data from all files
        data = []
        for data_file_name in files:
            with open(data_file_name, 'r') as data_file:
                data.append(yaml.load(data_file))
        # Interleave data
        in_data = []
        while data:
            # Randomly select an input stream
            idx = random.choice(range(len(data)))
            # Get next record
            in_data.append(data[idx].pop(0))
            # Check if list should be destroyed
            if not data[idx]:
                data.pop(idx)
        rospy.loginfo("Read {} data records from {} files.".format(len(in_data), len(files)))
    else:
        # Inform
        rospy.loginfo("Opening file for playback.".format(filename))
        # Read data from the file
        try:
            with open(filename, 'r') as data_file:
                in_data = yaml.load(data_file)
        except IOError:
            rospy.logerr("There was an error opening the log file.")
            return

    # Initialize publishers
    tuple_pub = rospy.Publisher('bum/tuple', Tuple, queue_size=10)
    ev_pub = rospy.Publisher('bum/evidence', Evidence, queue_size=10)

    # Wait for publishers to be ready
    rospy.sleep(2.)

    # Initialize empty messages
    tuple_msg = Tuple()
    ev_msg = Evidence()

    # Messages are publised 0.5 secs apart
    r = rospy.Rate(2)

    # For each entry in the log
    for record in in_data:
        if rospy.is_shutdown():
            break
        try:
            # If a characteristic is received, we publish as Tuple according to the GCD
            # Each different characteristic earns a different hard tuple publication
            # with its own evidence in the correct order
            for key in record["C"]:
                # Fill in easy fields
                tuple_msg.char_id = key
                tuple_msg.characteristic = record["C"][key]
                tuple_msg.user_id = record["Identity"]
                tuple_msg.h = 0.001
                tuple_msg.hard = True
                # Fill in evidence in the correct order according to the GCD
                ev = []
                for ev_id in GCD["C"][key]["input"]:
                    for key in record["Evidence"]:
                        if key == ev_id:
                            ev.append(record["Evidence"][key])
                tuple_msg.evidence = ev
                # Inform and publish
                rospy.loginfo("Publishing tuple: {}.".format(tuple_msg))
                tuple_pub.publish(tuple_msg)
        except KeyError:
            # If no characteristic is received, we publish as evidence    
            vals = []
            ids = []
            # Separate IDs and values to match message
            for key in record["Evidence"]:
                vals.append(record["Evidence"][key])
                ids.append(key)
            # Fill in message
            ev_msg.values = vals
            ev_msg.evidence_ids = ids
            ev_msg.user_id = record["Identity"]
            # Inform and publish
            rospy.loginfo("Publishing evidence: {}.".format(ev_msg))
            ev_pub.publish(ev_msg)        
    
        # Sleep for Rate
        r.sleep()


def tuple_callback(msg):
    """ A callback for directing tuples to the logging functions. 

    If a tuple is hard evidence, it is logged as evidence. If it is a soft
    tuple, it's logged as an execution."""
    # First we check whether we got hard evidence
    if msg.hard == True and log_hard == True:
        # In this case, we log as evidence
        # Get evidence IDs from the GCD
        ev_dict = dict()
        inputs = GCD["C"][msg.char_id]["input"]
        for i, ev_id in enumerate(inputs):
            ev_dict[ev_id] = msg.evidence[i]

        # Send to the logging function
        log_evidence(ev_dict, msg.user_id, msg.characteristic, msg.char_id, ev_log_file)
        pass
    else:
        # Otherwise, we log as a classification result
        log_classification(list(msg.evidence), msg.user_id, msg.characteristic, msg.char_id, msg.h, exec_log_file)


def evidence_callback(msg):
    """ A callback for directing evidence to the logging function. """
    # Evidence must be a dictionary
    ev_dict = dict()
    for i, ev_id in enumerate(msg.evidence_ids):
        ev_dict[ev_id] = msg.values[i]

    # Log this evidence
    log_evidence(ev_dict, msg.user_id, None, None, ev_log_file)


if __name__=="__main__":
    # Initialize ROS node
    rospy.init_node('data_manager_node')
    rospy.loginfo("BUM Data Manager ROS node started!")

    # Allocate variables
    ev_log_file = ""
    exec_log_file = ""
    gcd_filename = ""
    operation = "listen"

    # Get file names from parameters
    try:
        gcd_filename = rospy.get_param("bum_ros/gcd_file")
    except KeyError:
        rospy.logfatal("Could not get GCD file name parameter")
        exit()
    try:
        ev_log_file = rospy.get_param("bum_ros/ev_log_file")
    except KeyError:
        rospy.logfatal("Could not get evidence log file name parameter")
        exit()
    try:
        exec_log_file = rospy.get_param("bum_ros/exec_log_file")
    except KeyError:
        rospy.logfatal("Could not get exec log file name parameter")
        exit()

    # Get mode of operation from parameter
    try:
        operation = rospy.get_param("bum_ros/operation_mode")
    except KeyError:
        rospy.logwarn("Could not get mode of operation parameter, defaulting to listening mode.")

    # Read GCD file
    with open(gcd_filename, "r") as gcd_file:
        GCD = yaml.load(gcd_file)

    # Start operating
    if operation == "listen":
        rospy.loginfo("Entering listening mode.")
        # Initialize subscribers
        rospy.Subscriber("bum/tuple", Tuple, tuple_callback)
        rospy.Subscriber("bum/evidence", Evidence, evidence_callback)
        # Let it spin
        rospy.spin()
    elif operation == "playback":
        rospy.loginfo("Entering playback mode.")
        playback_evidence(ev_log_file)
    elif operation == "dual":
        rospy.loginfo("Entering dual mode.")
        # TODO: Improve
        log_hard = False
        rospy.Subscriber("bum/tuple", Tuple, tuple_callback)
        playback_evidence(ev_log_file)