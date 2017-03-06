#!/usr/bin/python3
""" This script implements the testbench used for producing the results for 
the paper.

T vector = [L (chosen label), E (evidence), h (entropy)]
"""
import probt
import numpy as np
import pprint

# This *dictionary* will maintain the characteristics of the users.
# It is indexed by the tuple (tuple(evidence) + tuple(identity)), and
# as such maintains the latest label for each combination of evidence
# and identity. Uninitialized characteristics are kept as -1
user_characteristics = []

class characteristic_model:
    """ This class contains a full model of a single characteristic.
    It is able to instantiate and fuse information, as well as to plot
    its likelihoods, etc.

    TODO: Normal usage
    """
    def __init__(self, input_classes=[10,10], output_classes=10, number_of_users=10):
        """ Initializes len(input_classes) input variables ranging from 1 to
        input_classes[i]. Initializes one output class ranging from 1 to
        output_class. Initializes the identity variable with number_of_users
        possible values.

        Also initializes the prior distribution (as uniform) as well as uniform
        initial likelihoods.
        """
        # Define Variables
        self._C_1 = probt.plVariable("C_1", probt.plIntegerType(1, output_classes))
        self._I = probt.plVariable("I", probt.plIntegerType(1, number_of_users))
        input_variables = [probt.plVariable("E_{}".format(i), probt.plIntegerType(1, input_classes[i])) for i in range(len(input_classes))]
        input_variables.append(self._I)
        self._input_variables = probt.plVariablesConjunction(input_variables)
        print(self._input_variables)

        # Define/fill in likelihood
        self._P_E_given_C_1 = probt.plDistributionTable(self._input_variables, self._C_1)
        values = probt.plValues(self._input_variables ^ self._C_1)
        for i in range(output_classes):
            values[self._C_1] = i+1    
            self._P_E_given_C_1.push(probt.plUniform(self._input_variables), values)

        # Generate prior (don't worry, probt normalizes it for us)
        self._P_C_1 = probt.plProbTable(self._C_1, [1]*output_classes)

        # Lists for keeping results
        self._in_sequence = []
        self._class_sequence = []
        self._prob_sequence = []
        self._identity_sequence = []


    def instantiate(self, evidence, identity):
        """ Instantiates the model, yielding a result class. """
        # Define evidence
        evidence_vals = probt.plValues(self._input_variables)
        for i in range(len(evidence)):
            evidence_vals[self._input_variables[i]] = evidence[i]
        # Identity is always the last input variable:
        evidence_vals[self._input_variables[len(evidence)]] = identity

        # Define joint distribution
        joint = probt.plJointDistribution(self._input_variables ^ self._C_1, self._P_C_1*self._P_E_given_C_1)
        
        # Define resulting conditional distribution
        P_C_1_given_E = joint.ask(self._C_1, self._input_variables)

        # Instantiate/Compile into resulting distribution
        result = P_C_1_given_E.instantiate(evidence_vals)
        result_class = result.n_best(1)[0].to_int()
        result_prob = result.tabulate()[1][result_class-1]

        # Append to report
        self._in_sequence.append(evidence)
        self._class_sequence.append(result_class)
        self._prob_sequence.append(result_prob)
        self._identity_sequence.append(identity)

        # Return result
        return result_class


    def fuse(self, label, evidence, identity, h):
        """ Updates the likelihood with a new example. 
        evidence: evidence vector
        label: class for which we're fusing
        h: entropy
        """
        # Initialize plValues
        values = probt.plValues(self._C_1)
        values[self._C_1] = label

        # Instantiate P(E|C_1=result)
        instantiation = self._P_E_given_C_1.instantiate(values).tabulate()
        values_vector = instantiation[0]
        dist = instantiation[1]
        
        # Reinforce the corresponding evidence
        # We use the plValues vector returned by tabulate to guide
        # the update process
        for i in range(len(values_vector)):
            elem = values_vector[i]
            # If the values correspond to the input values (looks REALLY arcane to be extensible)
            # basically, the set will be {True} if all variables in elem are equal to the inputs,
            # which is the case we want to reinforce.
            if set([True if elem[j] == evidence[j] else False for j in range(elem.size())]) == {True}:
                dist[i] = 2*dist[i]
        
        # Push into the distribution
        self._P_E_given_C_1.push(values, dist)


    def plot(label, evidence, identity):
        """ Plots the likelihood for the given evidence vector, identity label. """
        pass

    
    def report(self):
        """ Print history of (input, output, probability) tuples. """
        zipped = zip(self._in_sequence, self._identity_sequence, self._class_sequence, self._prob_sequence)
        print("(evidence, identity, output, probability)")
        pprint.pprint(list(zipped))

class user_simulator:
    """ A class for simulating a user. """
    pass

def generate_label(p_c, evidence, hard_label=-1):
    """ This function receives the P(C) distribution and the current
    evidence, and optionally a hard label.
    If the classification and hard label do not match up, the 
    generator has the chance to tech the system by incorporating the hard
    evidence into the T vector.

    This function also updates the global registry of users, for clustering purposes.

    returns T: as defined before
    hard_label: "correct" label received from elsewhere
    """
    pass

if __name__=="__main__":
    c1 = characteristic_model([10,10, 10], 10)
    
    for i in range(10):
        result_class = c1.instantiate([1, 2, 3], 1)
        #c1.fuse(result_class, [1, 2, 3], 1, 1.0)

    c1.report()