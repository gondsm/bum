#!/usr/bin/python
# Futures
from __future__ import division
from __future__ import print_function

# Standard Lib
import yaml

# ROS
import rospy
from bum_ros.msg import Likelihood, Tuple, Evidence
#from conductor import gmu_functions as robot

if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('bum_ros_conductor')

    # Initialize publishers
    tuple_pub = rospy.Publisher('bum/tuple', Tuple, queue_size=10)
    evidence_pub = rospy.Publisher('bum/evidence', Evidence, queue_size=10)

    rospy.sleep(3.)

    # Publish some hard evidence
    tuple_msg = Tuple()
    tuple_msg.char_id = "C1"
    tuple_msg.characteristic = 4
    tuple_msg.evidence = [1,2,3]
    tuple_msg.user_id = 1
    tuple_msg.h = 0.001
    tuple_msg.hard = True
    tuple_pub.publish(tuple_msg)

    rospy.sleep(3.)

    tuple_msg.char_id = "C2"
    tuple_msg.characteristic = 7
    tuple_msg.evidence = [1,2]
    tuple_msg.user_id = 1
    tuple_msg.h = 0.001
    tuple_msg.hard = True
    tuple_pub.publish(tuple_msg)

    rospy.sleep(3.)

    tuple_msg.char_id = "C3"
    tuple_msg.characteristic = 9
    tuple_msg.evidence = [3]
    tuple_msg.user_id = 1
    tuple_msg.h = 0.001
    tuple_msg.hard = True
    tuple_pub.publish(tuple_msg)

    rospy.sleep(3.)

    # Publish some regular evidence
    evidence_msg = Evidence()
    evidence_msg.values = [1,2,3]
    evidence_msg.evidence_ids = ["E1", "E2", "E3"]
    evidence_msg.user_id = 1
    evidence_pub.publish(evidence_msg)
