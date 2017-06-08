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

        # Read GDC file
        with open(gdc_filename, "r") as gdc_file:
            self._gdc = yaml.load(gdc_file)

        # Import configuration to local variables
        characteristics = self._gdc["C"]
        config = self._gdc["Config"]
        active = config["Active"]
        evidence = self._gdc["E"]
        nusers = self._gdc["nusers"]

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
                    self._tuple_pub.publish(out_msg)

        # Display predictions
        rospy.loginfo("New predictions:")
        rospy.loginfo(predictions)


    def tuple_callback(self, data):
        """ Receives a new tuple, which is fused into the model. """
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