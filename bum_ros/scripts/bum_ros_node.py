#!/usr/bin/python
""" The bum_ros_node script 
This is the main ROS node for the BUM system. It can instantiate and fuse
characteristics, as well as export and import likelihoods received from the 
remote system. It needs a GCD file to operate (see the config folder).
"""

# Copyright (C) 2017 University of Coimbra
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Original author and maintainer: Gonçalo S. Martins (gondsm@gmail.com)

# Futures
from __future__ import division
from __future__ import print_function

# Standard Lib
import yaml

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
        """ Initializes the ROS node and the estimators specified in the GCD. """
        # Initialize ROS node
        rospy.init_node('bum_ros_node')

        # Get GCD file from parameter
        gcd_filename = ""
        try:
            gcd_filename = rospy.get_param("bum_ros/gcd_file")
        except KeyError:
            rospy.logfatal("Could not get GCD file name parameter")
            return

        # Read GDC file
        with open(gcd_filename, "r") as gdc_file:
            self._gcd = yaml.load(gdc_file)

        # Import configuration to local variables
        characteristics = self._gcd["C"]
        config = self._gcd["Config"]
        active = config["Active"]
        evidence = self._gcd["E"]
        nusers = self._gcd["nusers"]

        # Initialize subscribers
        rospy.Subscriber("bum/likelihood", Likelihood, self.likelihood_callback)
        rospy.Subscriber("bum/evidence", Evidence, self.evidence_callback)
        if config["Fusion"]:
            rospy.Subscriber("bum/tuple", Tuple, self.tuple_callback)

        # Initialize publishers
        if config["Publish_on_predict"]:
            self._tuple_pub = rospy.Publisher('bum/tuple', Tuple, queue_size=10)
        else:
            # Initialize emtpy variable to signal that we won't publish
            self._tuple_pub = None

        # Instantiate objects according to GDC
        self._c_models = dict()
        for key in active:
            n_classes = characteristics[key]["nclasses"]
            evidence_structure = [evidence[ev]["nclasses"] for ev in characteristics[key]["input"]]
            self._c_models[key] = bum_classes.characteristic_model(evidence_structure, n_classes, nusers)

        # Report on the abilities of this node
        rospy.loginfo("BUM ROS node started!")
        rospy.loginfo("We have the following models:")
        for key in self._c_models:
           rospy.loginfo("\tCharacteristic {} with inputs {}.".format(key, self.get_characteristic_input(key)))
        rospy.loginfo("Fusion is active in this node: {}.".format(config["Fusion"]))
        rospy.loginfo("Publishing on estimation is active in this node: {}.".format(bool(self._tuple_pub)))


    def get_characteristic_input(self, c):
        """ Returns a list of the characteristc's input's ID strings. """
        return self._gcd["C"][c]["input"]


    def evidence_callback(self, data):
        """ Receives evidence and produces a new prediction. """
        rospy.loginfo("Received evidence!")
        # Import evidence to local variables
        vals = data.values
        ids = data.evidence_ids
        uid = data.user_id

        # Final result will be a dict, of course
        predictions = dict()

        # Predict each characteristic that is possible with this evidence
        for key in self._c_models:
            # Get the input IDs of the current model
            inputs = self.get_characteristic_input(key)

            # Now THIS is a comprehension
            # This gets the correct inputs for each characteristic according 
            # to its input variables, including the correct order.
            # This is the kind of thing you do just because you can.
            # A more readable version would include two fors, appending the
            # values from the input according to the codes in the
            # characteristic input values.
            char_input =  [item for sublist in [[vals[i] for i,x in enumerate(vals) if ids[i] == ev_id] for ev_id in inputs] for item in sublist]
            
            # If we have all of the inputs, we predict
            if len(char_input) == len(inputs):
                # Formalize evidence (with user ID)
                result, entropy = self._c_models[key].instantiate(char_input, uid)
                # If a prediction is made
                predictions[key] = result

                # Publish new tuple, if appropriate
                if self._tuple_pub:
                    out_msg = Tuple()
                    out_msg.char_id = key
                    out_msg.characteristic = result
                    out_msg.evidence = char_input
                    out_msg.user_id = uid
                    out_msg.h = entropy
                    out_msg.hard = False
                    self._tuple_pub.publish(out_msg)

        # Display predictions
        rospy.loginfo("New predictions:")
        rospy.loginfo(predictions)


    def tuple_callback(self, data):
        """ Receives a new tuple, which is fused into the model. """
        # Check if we should be here
        if self._gcd["Config"]["Only_fuse_hard"] and data.hard or not self._gcd["Config"]["Only_fuse_hard"]:
            rospy.loginfo("Received tuple!")
            # Import evidence to local variables for clarity
            vals = list(data.evidence)
            characteristic = data.characteristic
            char_id = data.char_id
            uid = data.user_id
            h = data.h

            # Fuse into the corresponding characteristic models
            T = [characteristic, vals, uid, h]
            self._c_models[char_id].fuse(T)
        else:
            rospy.loginfo("Received soft tuple, not fusing!")


    def likelihood_callback(self, data):
        """ Receives a likelihood, which replaces the existing likelihoods. """
        rospy.loginfo("Received likelihood!")
        # Separate likelihoods according to the respective characteristic

        # Replace likelihoods one by one


    def run(self):
        """ Spin ROS, essentially. """
        rospy.spin()


if __name__=="__main__":
    # Initialize object (with a sample file for now)
    b = BumRosNode()

    # Let it do its thing
    b.run()