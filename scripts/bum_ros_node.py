#!/usr/bin/python
# Futures
from __future__ import division
from __future__ import print_function

# ROS
import rospy
import bum_ros.msg

# Custom
import bum_classes


if __name__=="__main__":
	rospy.init_node('bum_ros_node')
	rospy.loginfo("BUM ROS node started!")

	#bum_ros.msg.Tuple()
	#bum_ros.msg.Likelihood()

	rospy.spin()