#!/usr/bin/python
# -*- coding: utf-8 -*-
# Futures
from __future__ import division
from __future__ import print_function

# Standard Lib
import yaml
import random
import time

# ROS
import rospy
from bum_ros.msg import Likelihood, Tuple, Evidence
from conductor import gmu_functions as robot


# The volume steps we'll cycle through
volume_steps = [40, 50, 60, 70, 75]

# Variables for controlling the robot's "steps"
initial_step = 2 # The robot will start in the middle
max_step = 4 # of a scale going from 0 to 4

# The list of questions to be used in this test
questions = ["one",
             "two",
             "three",
             "four"]
questions_distance = ["Do you think I am speaking to you at the correct distance?"]
questions_volume = ["Do you think I should speak at a different volume?"]

# List of evidences of talkativeness gathered by the system
ev_talk = []


def send_talkativeness_evidence(words, talk_time, ev):
    """ Stores and sends talkativeness evidence according to the words and 
    speech time received by the robot. ev is the evidence vector we're
    working with.
    """
    # Send talkativeness evidence
    ev.append(len(words.split())/talk_time)
    evidence_msg = Evidence()
    evidence_msg.values = ev
    evidence_msg.evidence_ids = ["Et{}".format(i) for i in reversed(range(len(ev)))]
    evidence_msg.user_id = 1
    evidence_pub.publish(evidence_msg)
    rospy.loginfo("Publishing new talkativeness evidence:")
    rospy.loginfo(evidence_msg)


if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('bum_ros_conductor')

    # Initialze the robot's functions
    robot.init_functions()

    # Initialize publishers
    tuple_pub = rospy.Publisher('bum/tuple', Tuple, queue_size=10)
    evidence_pub = rospy.Publisher('bum/evidence', Evidence, queue_size=10)

    # Loop over all questions
    while questions:
        # Ask a question and receive answer
        rospy.loginfo("Asking a generic question...")
        words, talk_time = robot.ask_question(questions, replace=False, timeout=1)
        send_talkativeness_evidence(words, talk_time, ev_talk)

        # Ask about volume
        rospy.loginfo("Asking volume...")
        words, talk_time = robot.ask_question(questions_volume, timeout=1)
        send_talkativeness_evidence(words, talk_time, ev_talk)

        # Ask about distance
        rospy.loginfo("Asking distance...")
        words, talk_time = robot.ask_question(questions_distance, timeout=1)
        send_talkativeness_evidence(words, talk_time, ev_talk)

    #rospy.sleep(3.)

    # Publish some hard evidence
    # tuple_msg = Tuple()
    # tuple_msg.char_id = "C1"
    # tuple_msg.characteristic = 4
    # tuple_msg.evidence = [1,2,3]
    # tuple_msg.user_id = 1
    # tuple_msg.h = 0.001
    # tuple_msg.hard = True
    # tuple_pub.publish(tuple_msg)

    # rospy.sleep(3.)

    # tuple_msg.char_id = "C2"
    # tuple_msg.characteristic = 7
    # tuple_msg.evidence = [1,2]
    # tuple_msg.user_id = 1
    # tuple_msg.h = 0.001
    # tuple_msg.hard = True
    # tuple_pub.publish(tuple_msg)

    # rospy.sleep(3.)

    # tuple_msg.char_id = "C3"
    # tuple_msg.characteristic = 9
    # tuple_msg.evidence = [3]
    # tuple_msg.user_id = 1
    # tuple_msg.h = 0.001
    # tuple_msg.hard = True
    # tuple_pub.publish(tuple_msg)

    # rospy.sleep(3.)

    # # Publish some regular evidence
    # evidence_msg = Evidence()
    # evidence_msg.values = [1,2,3]
    # evidence_msg.evidence_ids = ["E1", "E2", "E3"]
    # evidence_msg.user_id = 1
    # evidence_pub.publish(evidence_msg)

    #robot.step_forward()
    #robot.step_forward()
    #robot.step_forward(True)
    #robot.step_forward(True)

    