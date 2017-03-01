#!/usr/bin/python3
""" This script implements the testbench used for producing the results for 
the paper.
"""
import probt
import numpy as np
import pprint
    

class characteristic_model:
    def __init__(self, input_classes=[10,10], output_classes=10):
        """ Initializes len(input_classes) input variables ranging from 1 to
        input_classes[i]. Initializes one output class ranging from 1 to
        output_class.

        Also initializes the prior distribution (as uniform) as well as uniform
        initial likelihoods.
        """
        # Define Variables
        #self._E_1 = probt.plVariable("E_1", probt.plIntegerType(1, 10))
        #self._E_2 = probt.plVariable("E_2", probt.plIntegerType(1, 10))
        self._C_1 = probt.plVariable("C_1", probt.plIntegerType(1, output_classes))
        #self._input_variables = probt.plVariablesConjunction([self._E_1,self._E_2])
        self._input_variables = probt.plVariablesConjunction([probt.plVariable("E_{}".format(i), probt.plIntegerType(1, input_classes[i])) for i in range(len(input_classes))])
        #print(self._input_variables)

        # Define/fill in likelihood
        self._P_E_given_C_1 = probt.plDistributionTable(self._input_variables, self._C_1)
        values = probt.plValues(self._input_variables ^ self._C_1)
        for i in range(output_classes):
            values[self._C_1] = i+1    
            self._P_E_given_C_1.push(probt.plUniform(self._input_variables), values)
        #print("Initial likelihood", self._P_E_given_C_1)

        # Generate prior (don't worry, probt normalizes it for us)
        self._P_C_1 = probt.plProbTable(self._C_1, [1]*10)
        #print("Prior", self._P_C_1)

        # Lists for keeping results
        self._in_sequence = []
        self._class_sequence = []
        self._prob_sequence = []

    def instantiate(self, in_val):
        """ Instantiates the model, yielding a result class. """
        # Define joint distribution
        joint = probt.plJointDistribution(self._input_variables ^ self._C_1, self._P_C_1*self._P_E_given_C_1)
        
        # Define resulting conditional distribution
        P_C_1_given_E = joint.ask(self._C_1, self._input_variables)
        
        # Define evidence
        evidence = probt.plValues(self._input_variables)
        for i in range(len(in_val)):
            evidence[self._input_variables[i]] = in_val[i]
        
        # Instantiate/Compile into resulting distribution
        result = P_C_1_given_E.instantiate(evidence)
        result_class = result.n_best(1)[0].to_int()
        result_prob = result.tabulate()[1][result_class-1]

        # Append to report
        self._in_sequence.append(in_val)
        self._class_sequence.append(result_class)
        self._prob_sequence.append(result_prob)

        return result_class

    def fuse(self, in_val, result_class):
        """ Updates the likelihood with a new example. """
        # Initialize plValues
        values = probt.plValues(self._C_1)
        values[self._C_1] = result_class

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
            if set([True if elem[j] == in_val[j] else False for j in range(elem.size())]) == {True}:
                dist[i] = 2*dist[i]
        
        # Push into the distribution
        self._P_E_given_C_1.push(values, dist)

    def define_likelihood(self, out_class, likelihood):
        """ Replaces the likelihood """
        values = probt.plValues(self._C_1)
        values[self._C_1] = out_class
        self._P_E_given_C_1.push(probt.plUniform(self._input_variables), values)

    def report(self):
        """ Print history of (input, output, probability) tuples. """
        zipped = zip(self._in_sequence, self._class_sequence, self._prob_sequence)
        print("(input, output, probability)")
        pprint.pprint(list(zipped))
        return self._class_sequence


if __name__=="__main__":
    c1 = characteristic_model([10,10, 10], 10)
    
    for i in range(10):
        result_class = c1.instantiate([1, 2, 3])
        c1.fuse([1, 2, 3], result_class)

    c1.report()