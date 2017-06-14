#!/usr/bin/python
# Futures
from __future__ import division
from __future__ import print_function

# Standard Lib
import yaml

# ROS
import rospy
from bum_ros.msg import Likelihood, Tuple, Evidence

GDC = None

def log_evidence(evidence, identity, characteristics=None, filename=None):
    """ This function adds a new record to the evidence log. 

    Charateristics are received as a dict "char_id" -> value. 
    Evidence is also received as a dict "ev_id" -> value
    """
    # Sanitize inputs
    if type(evidence) is not dict:
        rospy.logerr("Evidence vector should be a dict!")
        return
    if type(identity) is not int:
        rospy.logerr("Identity should be an integer!")
        return
    if characteristics and type(characteristics) is not dict:
        rospy.logerr("Characteristics vector should be a dict!")
        return

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
    if characteristics:
        data_dict["C"] = characteristics
    data.append(data_dict)

    # Write to file
    data_file.write(yaml.dump(data, default_flow_style=False))


def log_classification(evidence, identity, characteristic, char_id, entropy, filename):
    """ This function adds a new record to the execution log, for later
    evaluation.
    """
    # TODO: Sanitize inputs

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
    data_dict[char_id] = characteristic
    data.append(data_dict)

    # Write to file
    data_file.write(yaml.dump(data, default_flow_style=False))


def playback_evidence(filename):
    """ This funcion plays back evidence recorded in filename. 

    For each entry in the log, if it has a characteristic associated, it is
    interpreted as hard evidence and published as a tuple message. If it does
    not, it is published as simple Evidence message.

    Messages are published 0.5 secs apart.
    """
    rospy.loginfo("Playing back evidence")

    # Read file
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
        try:
            # If a characteristic is received, we publish as Tuple according to the GDC
            # Each different characteristic earns a different hard tuple publication
            # with its own evidence in the correct order
            for key in record["C"]:
                # Fill in easy fields
                tuple_msg.char_id = key
                tuple_msg.characteristic = record["C"][key]
                tuple_msg.user_id = record["Identity"]
                tuple_msg.h = 0.001
                tuple_msg.hard = True
                # Fill in evidence in the correct order according to the GDC
                ev = []
                for ev_id in GDC["C"][key]["input"]:
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


if __name__=="__main__":
    # Initialize ROS node
    rospy.init_node('data_manager_node')
    rospy.loginfo("BUM Data Manager ROS node started!")

    # TODO: Receive these as input
    ev_log_file = "/home/vsantos/catkin_ws/src/bum_ros/config/ev_log.yaml"
    exec_log_file = "/home/vsantos/catkin_ws/src/bum_ros/config/exec_log.yaml"
    gdc_filename = "/home/vsantos/catkin_ws/src/bum_ros/config/caracteristics.gcd"
    
    # Read GDC file
    with open(gdc_filename, "r") as gdc_file:
        GDC = yaml.load(gdc_file)

    #log_evidence({"E1": 1, "E2": 2, "E3": 3},2, {"C1": 2}, filename=ev_log_file)
    #log_evidence({"E1": 1, "E2": 2, "E3": 3},2, filename=ev_log_file)
    #log_evidence({"E1": 1, "E2": 2, "E3": 3},2, filename=ev_log_file)
    #log_evidence({"E1": 1, "E2": 2, "E3": 3},2, filename=ev_log_file)
    #playback_evidence(ev_log_file)
    #log_classification([1,2,3], 2, 1, "C1", 0.1, exec_log_file)

    rospy.spin()