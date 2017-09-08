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


# Global Variables: The truth is that most, if not all, could be in the main function.
# Signals whether the answers should be retrieved from speech recognition or entered manually
keyboard_mode = True

# The volume steps we'll cycle through
volume_steps = [40, 50, 60, 70, 75]
current_volume = 2 # Starts in the middle of the scale

# Variables for controlling the robot's "steps"
current_step = 2 # The robot will start in the middle
max_step = 4 # of a scale going from 0 to 4

# The list of questions to be used in this test
questions = ["What can you tell me about your day?",
             "What are your favourite hobbies?",
             "How is your research going? Are you getting good results?",
             "Did you study electrical engineering? What did you like the most about it?",
             "Being a robot, I feel thunderstorms very personally. How do you feel about the weather we have been having lately?"]
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

# Have we already asked about volume and distance?
asked_distance = False
asked_volume = False

def send_talkativeness_evidence(words, talk_time, ev):
    """ Stores and sends talkativeness evidence according to the words and 
    speech time received by the robot. ev is the evidence vector we're
    working with.

    This evidence varies in the range [0, 20[, where 19 represents the maximum
    of 300 words per minute, or 5 words per second.
    """
    # Send talkativeness evidence
    new_ev = len(words.split())/talk_time
    new_ev = int((19/5)*new_ev)
    ev.append(new_ev)
    evidence_msg = Evidence()
    evidence_msg.values = ev
    evidence_msg.evidence_ids = ["Et{}".format(i) for i in reversed(range(len(ev)))]
    evidence_msg.user_id = 1
    rospy.loginfo("Publishing new talkativeness evidence.")
    #rospy.loginfo(evidence_msg)
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
    rospy.loginfo("Publishing new volume tuple.")
    #rospy.loginfo(tuple_msg)
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
    rospy.loginfo("Publishing new distance tuple.")
    #rospy.loginfo(tuple_msg)
    tuple_pub.publish(tuple_msg)


if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('bum_ros_conductor')

    # Initialze the robot's functions
    robot.init_functions()

    # Gather ground truth
    if len(sys.argv) > 1 and sys.argv[1] == "gt":
        val = []
        while val != "q":
            rospy.loginfo("Enter ground truth command [fw, rw, number]. q to exit:")
            val = raw_input()
            if val == "fw":
                # Take a step forward
                robot.step_forward()
            if val == "rw":
                # Take a step back
                robot.step_forward(reverse=True)
            try:
                # Speak at the given level
                robot.change_volume(volume_steps[int(val)])
                robot.speak("I am now speaking for a test.", lang="en-EN")
            except ValueError:
                pass
        exit()

    # Initialize publishers
    tuple_pub = rospy.Publisher('bum/tuple', Tuple, queue_size=10)
    evidence_pub = rospy.Publisher('bum/evidence', Evidence, queue_size=10)

    # Initialize correct volume
    robot.change_volume(volume_steps[current_volume])

    # Loop over all questions
    for i in range(6):
        # Ask a question and receive answer
        rospy.loginfo("Asking a generic question...")
        questions.pop(0)
        #words, talk_time = robot.ask_question(questions, replace=False, speech_time=True, keyboard_mode=keyboard_mode)
        #send_talkativeness_evidence(words, talk_time, ev_talk)

        # Ask about volume if we haven't already
        if not asked_volume and (random.random() < 0.25 or i > 3):
            # Ask about volume
            rospy.loginfo("Asking volume...")
            # Now we've asked volume
            asked_volume = True
            words, talk_time = robot.ask_question(questions_volume, speech_time=True, keyboard_mode=keyboard_mode)
            send_talkativeness_evidence(words, talk_time, ev_talk)
            # Process volume response in a loop until the user says they are happy with the volume.
            if(re.search(re_no, words) or re.search(re_louder, words) or re.search(re_quieter, words)):
                while not re.search(re_yes, words):
                    if(re.search(re_louder, words)):
                        if current_volume < len(volume_steps)-1:
                            current_volume += 1
                            robot.change_volume(volume_steps[current_volume])
                            robot.speak("Okay, I'll increase my volume a bit.", lang="en-EN")
                        else:
                            robot.speak("I'm sorry, I cannot increase the volume any further.", lang="en-EN")
                    elif(re.search(re_quieter, words)):
                        if current_volume > 0:
                            current_volume -= 1
                            robot.change_volume(volume_steps[current_volume])
                            robot.speak("Okay, I'll lower my volume a bit.", lang="en-EN")
                        else:
                            robot.speak("I'm sorry, I cannot lower the volume any further.", lang="en-EN")
                    elif(re.search(re_no, words)):
                        words, talk_time = robot.ask_question("Should I be speaking louder or with a lower tone of voice?", speech_time=True, keyboard_mode=keyboard_mode)
                        send_talkativeness_evidence(words, talk_time, ev_talk)
                        continue
                    words, talk_time = robot.ask_question("Is the volume okay now?", speech_time=True, keyboard_mode=keyboard_mode)
                    send_talkativeness_evidence(words, talk_time, ev_talk)
                    # Send tuple we got
                    send_volume_tuple(current_volume)
                robot.speak("Okay, I'll keep this volume.", lang="en-EN")
            else:
                # Send tuple we got
                send_volume_tuple(current_volume)
                robot.speak("Okay, I'll keep this volume.", lang="en-EN")
            # ... and we skip to the next question
            continue

        # Ask about distance if we haven't already
        if not asked_distance and (random.random() < 0.25 or i > 3):
            # Ask about distance
            rospy.loginfo("Asking distance...")
            # Now we've asked distance
            asked_distance = True
            words, talk_time = robot.ask_question(questions_distance, speech_time=True, keyboard_mode=keyboard_mode)
            send_talkativeness_evidence(words, talk_time, ev_talk)
            # Process volume response in a loop until the user says they are happy with the volume.
            if(re.search(re_no, words) or re.search(re_farther, words) or re.search(re_closer, words)):
                while not re.search(re_yes, words):
                    if(re.search(re_farther, words)):
                        if current_step < max_step-1:
                            current_step += 1
                            robot.step_forward(reverse=True)
                            robot.speak("Okay, I'll get a bit farther from you.", lang="en-EN")
                        else:
                            robot.speak("I'm sorry, I cannot go further back.", lang="en-EN")
                    elif(re.search(re_closer, words)):
                        if current_step > 0:
                            current_step -= 1
                            robot.step_forward()
                            robot.speak("Okay, I'll get a bit closer to you.", lang="en-EN")
                        else:
                            robot.speak("I'm sorry, I cannot go further forward.", lang="en-EN")
                    elif(re.search(re_no, words)):
                        words, talk_time = robot.ask_question("Should I be speaking closer or farther away from you?", speech_time=True, keyboard_mode=keyboard_mode)
                        send_talkativeness_evidence(words, talk_time, ev_talk)
                        continue
                    words, talk_time = robot.ask_question("Is the distance okay now?", speech_time=True, keyboard_mode=keyboard_mode)
                    send_talkativeness_evidence(words, talk_time, ev_talk)
                    # Send the tuple we got
                    send_distance_tuple(current_step)
                robot.speak("Okay, I'll keep this distance from you.", lang="en-EN")
            else:
                # Send the tuple we got
                send_distance_tuple(current_step)
                robot.speak("Okay, I'll keep this distance from you.", lang="en-EN")
            # ... and we skip to the next question
            continue

    robot.speak("I am done asking you questions now. Thank you for participating in this study.", lang="en-EN")
   