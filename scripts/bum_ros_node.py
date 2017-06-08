#!/usr/bin/python
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
    def __init__(self, gdc_filename):
        """ Initializes the ROS node and the estimators specified. """
        # Initialize ROS node
        rospy.init_node('bum_ros_node')
        rospy.loginfo("BUM ROS node started!")

        # Initialize subscribers
        rospy.Subscriber("bum/likelihood", Likelihood, self.likelihood_callback)
        rospy.Subscriber("bum/evidence", Evidence, self.evidence_callback)
        rospy.Subscriber("bum/tuple", Tuple, self.tuple_callback)

        # Read Global Characteristic Description
        with open(gdc_filename, "r") as gdc_file:
            self._gdc = yaml.load(gdc_file)
            #print(self._gdc)

        # Instantiate objects according to GDC
        characteristics = self._gdc["C"]
        active = self._gdc["Active"]
        evidence = self._gdc["E"]
        self._c_models = dict()
        for key in active:
            n_classes = characteristics[key]["nclasses"]
            evidence_structure = [evidence[ev]["nclasses"] for ev in characteristics[key]["input"]]
            nusers = 10 #TODO: Get this from somewhere
            self._c_models[key] = bum_classes.characteristic_model(evidence_structure, n_classes, nusers)

        #for key in self._c_models:
        #   rospy.loginfo("We have models for {}.".format(key))
        #   rospy.loginfo("With inputs {}.".format(self.get_characteristic_input(key)))


    def get_characteristic_input(self, c):
        """ Returns a list of the characteristc's input's ID strings. """
        return self._gdc["C"][c]["input"]


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
                result = self._c_models[key].instantiate(char_input, uid)
                # If a prediction is made
                predictions[key] = result

        # Return predictions
        rospy.loginfo("New predictions:")
        rospy.loginfo(predictions)


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
    # Initialize object (with a sample file for now)
    b = BumRosNode("/home/vsantos/catkin_ws/src/bum_ros/config/caracteristics.gcd")

    # Let it do its thing
    b.run()