#!/usr/bin/python
# -*- coding: utf-8 -*-
# Futures
from __future__ import division
from __future__ import print_function

# Standard Lib
import yaml
import random
import time
import re
import sys

# ROS
import rospy
from bum_ros.msg import Likelihood, Tuple, Evidence
from conductor import gmu_functions as robot


# The volume steps we'll cycle through
volume_steps = [40, 50, 60, 70, 75]
current_volume = 2 # Starts in the middle of the scale

# Variables for controlling the robot's "steps"
current_step = 2 # The robot will start in the middle
max_step = 4 # of a scale going from 0 to 4

# The list of questions to be used in this test
questions = ["one",
             "two",
             "three",
             "four"]
questions_distance = ["Do you think I am speaking to you at the correct distance?"]
questions_volume = ["Do you think I am speaking at the correct volume?"]

# List of evidences of talkativeness gathered by the system
ev_talk = []

# Regexes for answer processing
re_yes = ".*(yes|of course).*"
re_no = ".*(no|not).*"
re_louder = ".*(high|loud).*"
re_quieter = ".*(quiet|low).*"
re_closer = ".*(close|near).*"
re_farther = ".*(away|far).*"

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
    rospy.loginfo("Publishing new talkativeness evidence:")
    rospy.loginfo(evidence_msg)
    evidence_pub.publish(evidence_msg)


def send_volume_tuple(val):
    """ Sends a volume tuple of value val. """
    tuple_msg = Tuple()
    tuple_msg.char_id = "C1"
    tuple_msg.characteristic = val
    tuple_msg.evidence = []
    tuple_msg.user_id = 1
    tuple_msg.h = 0.001
    tuple_msg.hard = True
    rospy.loginfo("Publishing new volume tuple:")
    rospy.loginfo(tuple_msg)
    tuple_pub.publish(tuple_msg)


def send_distance_tuple(val):
    """ Sends a distance tuple of value val. """
    tuple_msg = Tuple()
    tuple_msg.char_id = "C2"
    tuple_msg.characteristic = val
    tuple_msg.evidence = []
    tuple_msg.user_id = 1
    tuple_msg.h = 0.001
    tuple_msg.hard = True
    rospy.loginfo("Publishing new distance tuple:")
    rospy.loginfo(tuple_msg)
    tuple_pub.publish(tuple_msg)


def manual_transcription(query, kw_repeat=["repeat", "not understand"], lang="en-EN", txt_repeat="I will repeat.", replace=True, timeout=20, speech_time=False):
    """ Emulates the speech recog function with raw_input. """
    if type(query) is not list:
        rospy.loginfo("Asking question: \"{}\".".format(query))
    else:
        rospy.loginfo("Asking question: \"{}\".".format(query[0]))
    init = time.time()
    text = raw_input().decode(sys.stdin.encoding or locale.getpreferredencoding(True))
    talk_time = time.time() - init

    return text, talk_time


if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('bum_ros_conductor')

    # Initialze the robot's functions
    robot.init_functions()

    # Initialize publishers
    tuple_pub = rospy.Publisher('bum/tuple', Tuple, queue_size=10)
    evidence_pub = rospy.Publisher('bum/evidence', Evidence, queue_size=10)

    # Switch into transcription mode
    #robot.ask_question = manual_transcription

    # Loop over all questions
    while questions:
        # Ask a question and receive answer
        rospy.loginfo("Asking a generic question...")
        words, talk_time = robot.ask_question(questions, replace=False)
        send_talkativeness_evidence(words, talk_time, ev_talk)

        # Ask about volume
        rospy.loginfo("Asking volume...")
        words, talk_time = robot.ask_question(questions_volume)
        send_talkativeness_evidence(words, talk_time, ev_talk)
        if(re.search(re_no, words)):
            words, talk_time = robot.ask_question("Should I be speaking louder or with a lower tone of voice?")
            send_talkativeness_evidence(words, talk_time, ev_talk)
            if(re.search(re_louder, words)):
                if current_volume < len(volume_steps)-1:
                    current_volume += 1
                    robot.change_volume(volume_steps[current_volume])
                else:
                    robot.speak("I'm sorry, I cannot increase the volume any further.", lang="en-EN")
            elif(re.search(re_quieter, words)):
                if current_volume > 0:
                    current_volume -= 1
                    robot.change_volume(volume_steps[current_volume])
                else:
                    robot.speak("I'm sorry, I cannot lower the volume any further.", lang="en-EN")
        send_volume_tuple(current_volume)

        # Ask about distance
        rospy.loginfo("Asking distance...")
        words, talk_time = robot.ask_question(questions_distance)
        send_talkativeness_evidence(words, talk_time, ev_talk)
        if(re.search(re_no, words)):
            words, talk_time = robot.ask_question("Should I be closer or farther away from you?")
            send_talkativeness_evidence(words, talk_time, ev_talk)
            if(re.search(re_farther, words)):
                if current_step < max_step-1:
                    current_step += 1
                    robot.step_forward(reverse=True)
                else:
                    robot.speak("I'm sorry, I cannot go further back.", lang="en-EN")
            elif(re.search(re_closer, words)):
                if current_step > 0:
                    current_step -= 1
                    robot.step_forward()
                else:
                    robot.speak("I'm sorry, I cannot go further forward.", lang="en-EN")
        send_distance_tuple(current_step)
   