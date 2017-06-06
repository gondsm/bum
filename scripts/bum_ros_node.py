#!/usr/bin/python
# Futures
from __future__ import division
from __future__ import print_function

# ROS
import rospy
from bum_ros.msg import Likelihood, Tuple, Evidence

# Custom
import bum_classes


class BumRosNode:
	""" This class is responsible for maintaining the whole BUM "device",
	including the various characteristic estimators, their likelihoods,
	and so on. It responds to a number of ROS topics according to the
	design of the BUM system.
	"""
	def __init__(self):
		""" Initializes the ROS node and the estimators specified. """
		# Initialize ROS node
		rospy.init_node('bum_ros_node')
		rospy.loginfo("BUM ROS node started!")

		# Initialize subscribers
		rospy.Subscriber("bum/likelihood", Likelihood, self.likelihood_callback)
		rospy.Subscriber("bum/evidence", Evidence, self.evidence_callback)
		rospy.Subscriber("bum/tuple", Tuple, self.tuple_callback)

		# TODO: Read Global Characteristic Description


	def evidence_callback(self, data):
		""" Receives evidence and produces a new prediction. """
		rospy.loginfo("Received evidence!")
		# Separate evidence according to the characteristic they correspond to

		# Predict each characteristic that is possible with this evidence

		# Return predictions


	def tuple_callback(self, data):
		""" Receives a new tuple, which is fused into the model. """
		rospy.loginfo("Received tuple!")
		# Separate tuple according to the characteristic

		# Fuse into each of the characteristic models


	def likelihood_callback(self, data):
		""" Receives a likelihood, which replaces the existing likelihoods. """
		rospy.loginfo("Received likelihood!")
		# Separate likelihoods according to the respective characteristic

		# Replace likelihoods one by one


	def run(self):
		""" Spin ROS, essentially. """
		rospy.spin()


if __name__=="__main__":
	# Initialize object
	b = BumRosNode()

	# Let it do its thing
	b.run()