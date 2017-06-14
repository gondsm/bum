#!/usr/bin/python
# Futures
from __future__ import division
from __future__ import print_function

# Standard Lib
import yaml

# ROS
import rospy
from bum_ros.msg import Likelihood, Tuple, Evidence




if __name__=="__main__":
    # Initialize ROS node
    rospy.init_node('data_manager_node')
    rospy.loginfo("BUM Data Manager ROS node started!")