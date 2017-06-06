#!/usr/bin/python
# Futures
from __future__ import division
from __future__ import print_function

# ROS
import rospy

# Custom
import bum_classes


if __name__=="__main__":
	rospy.init_node('bum_ros_node')
	rospy.loginfo("BUM ROS node started!")
	rospy.spin()