""" The bum_classes script
This script contains the main classes to be used by the BUM system, namely the
characteristics modules, based on the ProBT computational engine, and the
population simulator used for testing the system.
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

# Standard
import pprint
import itertools
import copy
import os
import pickle
from multiprocessing import Pool
from multiprocessing import Process
import time

# Numpy
import numpy as np
import numpy.random as rnd

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

# Scikit-learn
from sklearn import mixture

# Scipy
from scipy import linalg

# Custom
import probt

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
        self._C_1 = probt.plVariable("C_1", probt.plIntegerType(0, output_classes-1))
        self._I = probt.plVariable("I", probt.plIntegerType(0, number_of_users-1))
        input_variables = [probt.plVariable("E_{}".format(i), probt.plIntegerType(0, input_classes[i]-1)) for i in range(len(input_classes))]
        input_variables.append(self._I)
        self._input_variables = probt.plVariablesConjunction(input_variables)

        # Define/fill in likelihood
        self._P_E_given_C_1 = probt.plDistributionTable(self._input_variables, self._C_1)
        values = probt.plValues(self._input_variables ^ self._C_1)
        for i in range(output_classes):
            values[self._C_1] = i  
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
        result = P_C_1_given_E.compile().instantiate(evidence_vals)
        result_class = result.n_best(1)[0].to_int()
        result_prob = result.tabulate()[1][result_class]
        entropy = result.compute_shannon_entropy()
        entropy = 0.1 if entropy < 0.1 else entropy

        # Append to report
        self._in_sequence.append(evidence)
        self._class_sequence.append(result_class)
        self._prob_sequence.append(result_prob)
        self._identity_sequence.append(identity)

        # Return result
        return [result_class, entropy]


    def fuse(self, T):    
        """ Updates the likelihood with a new example. 
        The T vector is received as defined above.
        """
        # Break up the T vector:
        label = T[0]
        evidence = T[1]
        identity = T[2]
        h = T[3]

        # Initialize plValues
        values = probt.plValues(self._C_1)
        values[self._C_1] = label

        # Initialize temporary input variable (so we can compare everything as
        # vectors and not worry too much)
        in_vals = evidence + [identity]

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
            if set([True if elem[j] == in_vals[j] else False for j in range(elem.size())]) == {True}:
                if dist[i] < 0.80:
                    dist[i] = (1+1/h)*dist[i]
        
        # Push into the distribution
        self._P_E_given_C_1.push(values, dist)


    def get_likelihood(self):
        """ Gets the likelihood of the model, so that it can be transmitted
        elsewhere and used by other models."""
        return self._P_E_given_C_1.tabulate()

    def set_likelihood(self, likelihood):
        """ Sets the likelihood of the system to that obtained by another
        model."""
        self._P_E_given_C_1.replace(self._C_1, self._input_variables, likelihood)

    def plot(self, label, evidence, identity):
        """ Plots the likelihood for the given evidence vector, identity label. """
        # Initialize plValues
        values = probt.plValues(self._C_1)
        values[self._C_1] = label

        # Instantiate P(E|C_1=result)
        instantiation = self._P_E_given_C_1.instantiate(values)
        print(instantiation)

    
    def report(self):
        """ Print history of (input, output, probability) tuples. """
        zipped = zip(self._in_sequence, self._identity_sequence, self._class_sequence, self._prob_sequence)
        print("(evidence, identity, output, probability)")
        pprint.pprint(list(zipped))


class population_simulator:
    """ A class for simulating a user population. """

    def __init__(self, number_of_users=10, evidence_structure=[10,10,10], characteristics_structure=[10], profiles=None):
        """ Initialize a pool of number_of_users users.
        It receives characteristics and evidence structures. These are lists that
        maintain the number of states of each variable and, implicitly, how 
        many of each there are.
        """
        # This *dictionary* will maintain the characteristics of the users.
        # It is indexed by the tuple (tuple(evidence) + tuple(identity)), and
        # as such maintains the label for each combination of evidence
        # and identity. It is similar to the structure used for keeping the
        # results. This is by design; they can then be compared.
        self._users = dict()
        
        # Copy input to members:
        self._number_of_users = number_of_users
        self._evidence_structure = evidence_structure
        self._characteristics_structure = characteristics_structure
        self._profiles = profiles

        # Generate population
        if profiles is None:
            self.__generate_uniform_population()
        else:
            self.__generate_from_profiles()


    def __generate_uniform_population(self):
        """ Generates a population without set profiles using uniform distributions. """
        # Build the ranges for all evidence and create an iterator
        a = [range(0, elem) for elem in self._evidence_structure]

        # Initialize characteristics for all evidence combination
        for i in range(self._number_of_users):
            iterator = itertools.product(*a)
            for comb in iterator:
                self._users[comb + (i,)] = [np.random.randint(0, elem) for elem in self._characteristics_structure]


    def __generate_from_profiles(self):
        """ Generates a population from a set of given profiles. """
        #print("Generating guys from profiles", profiles)
        # Build the ranges for all evidence and create an iterator
        a = [range(0, elem) for elem in self._evidence_structure]

        # Initialize characteristics for all evidence combination uniformly
        for i in range(self._number_of_users):
            iterator = itertools.product(*a)
            for comb in iterator:
                self._users[comb + (i,)] = [np.random.randint(0, elem) for elem in self._characteristics_structure]
        
        # Re-initialize some evidence according to the profiles
        for key in self._profiles:
            # Attribute each user to a profile
            attribution = np.random.randint(len(self._profiles[key]), size=self._number_of_users)

            # Generate actual stuff
            for i, p in enumerate(attribution):
                # Get characteristics in selected profile
                # we have to copy, otherwise stuff gets funky
                # when profiles gets destroyed.
                charac = copy.copy(self._profiles[key][p])

                # Add noise
                for j in range(len(charac)):
                    charac[j] += np.random.randint(-1, 2)
                    if charac[j] < 0:
                        charac[j] = 0
                    if charac[j] >= self._characteristics_structure[j]:
                        charac[j] = self._characteristics_structure[j]-1
                #print(charac)

                # Put into user population
                self._users[key + (i,)] = charac


    def generate(self, user=None):
        """ Generates a combination of evidence and labels, according to
        the profiles loaded in self._users.
        If a user ID is received, it is used to retrieve evidence from that user.
        """
        # Generate evidence
        evidence = [np.random.randint(0, elem) for elem in self._evidence_structure]
        if user is not None:
            evidence.append(user)
        else:
            evidence.append(np.random.randint(0, self._number_of_users))

        # Retrieve characteristics
        characteristics = self._users[tuple(evidence)]

        # Return a list containing both
        return evidence, characteristics


    def get_users(self):
        """ Returns the whole users dictionary, which can then be used by
        other pieces of code for evaluation.
        """
        # Return the user dictionary
        return self._users