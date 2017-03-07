#!/usr/bin/python3
""" This script implements the testbench used for producing the results for 
the paper.

T vector = [L (chosen label), E (evidence), h (entropy)]
"""

# Futures
# This was tested on Python3, but might also work on 2.7.
# The following is DEFINITELY needed:
from __future__ import division
from __future__ import print_function

# Standard imports
import numpy as np
import pprint
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

# Custom imports
import probt


# Constants
# These are global constant variables that guide the instantiation of the
# various objects.
# How many users we'll simulate:
NUMBER_OF_USERS = 10
# How many different values are in each evidence variable:
# (implicitly defines the name and number of evidence variables)
EVIDENCE_STRUCTURE = [10, 10, 10] 
# How many different values are in the output variables:
# (implicitly defined the name and number of output variables)
CHARACTERISTICS_STRUCTURE = [10]
# How many iterations the system will run for:
NUMBER_OF_ITERATIONS = 100

# This *dictionary* will maintain the characteristics of the users.
# It is indexed by the tuple (tuple(evidence) + tuple(identity)), and
# as such maintains the latest label for each combination of evidence
# and identity. No initialization will be done; uninitialized values
# will raise a KeyError.
user_characteristics = dict()


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
        result = P_C_1_given_E.compile().instantiate(evidence_vals)
        result_class = result.n_best(1)[0].to_int()
        result_prob = result.tabulate()[1][result_class-1]
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

    def __init__(self, number_of_users=10, evidence_structure=[10,10,10], characteristics_structure=[10]):
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

        # Build the ranges for all evidence and create an iterator
        a = [range(1, elem+1) for elem in evidence_structure]

        # Initialize characteristics for all evidence combination
        for i in range(number_of_users):
            iterator = itertools.product(*a)
            for comb in iterator:
                self._users[comb + (i+1,)] = [np.random.randint(1, elem+1) for elem in characteristics_structure]


    def generate(self, user=-1):
        """ Generates a combination of evidence and labels, according to
        the profiles loaded in self._users.
        """
        # Generate evidence
        evidence = [np.random.randint(1, elem+1) for elem in self._evidence_structure]
        evidence.append(np.random.randint(1, self._number_of_users+1))

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


def generate_label(soft_label, entropy, evidence, identity, hard_label=None):
    """ This function receives the soft_label and entropy and the current
    evidence, and optionally a hard label.
    If the classification and hard label do not match up, the 
    generator has the chance to tech the system by incorporating the hard
    evidence into the T vector.

    This function also updates the global registry of users, for clustering purposes.

    It may seem like a really simple function, but in the case where the system
    is distributed, this function generates the main piece of data that is
    transmitted between the local and remote parts of the system.

    returns T: as defined before
    soft_label: classification result
    entropy: entropy of the result distribution
    evidence: the evidence vector
    identity: the user's identity
    hard_label: "correct" label received from elsewhere
    """

    # Define T with soft label
    T = [soft_label, evidence, identity, entropy]

    try:
        # If the value was initialized:
        user_characteristics[tuple(T[1] + [T[2]])]
    except KeyError:
        # Define T vector with hard evidence, if possible
        if hard_label is not None:
            T = [hard_label, evidence, identity, 0.01]

    # Update user representation
    user_characteristics[tuple(T[1] + [T[2]])] = [soft_label]

    # Return the vector for fusion
    return T


def calculate_accuracy(learned_dict, reference_dict):
    """ This function calculates the classification accuracy given the learned
    characteristics of the user.
    """
    count = 0
    correct = 0
    for key, val in learned_dict.items():
        count += 1
        if val == reference_dict[key]:
            correct += 1

    accuracy = (correct/count)

    #print("Count:", count, "correct:", correct, "accuracy:", accuracy*100, "%.")

    return [accuracy, count, correct]


def plot_users(users, evidence):
    # Set mpl parameters
    mpl.rcParams['legend.fontsize'] = 10
    #mpl.rcParams['text.usetex'] = True

    colors=["blue", "red", "orange"]

    # Initialize  and adjust figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_zlim([0,10])
    ax.set_xlabel("C_1")
    ax.set_ylabel("C_2")
    ax.set_zlabel("C_3")
    ax.view_init(elev=15., azim=45)

    # Plot/Save
    plt.hold(True)
    for i, user in enumerate(users):
        # Point
        ax.plot([user[0]], [user[1]], [user[2]], 'o', label='User {}'.format(i+1))
        
    #ax.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
    #ax.legend(numpoints=1, ncol=3)
    plt.show()
    #plt.savefig("users_example.pdf")


def iterative_test():
    """ The "regular" test that is run with this script. 
    
    TODO: What does it do?
    """
    # Initialize population simulator
    population = population_simulator(NUMBER_OF_USERS, EVIDENCE_STRUCTURE, CHARACTERISTICS_STRUCTURE)

    # Define models
    c1 = characteristic_model(EVIDENCE_STRUCTURE, CHARACTERISTICS_STRUCTURE[0], NUMBER_OF_USERS)

    # List to maintain the accuracy of the system
    accuracy = []

    # Run
    for i in range(NUMBER_OF_ITERATIONS):
        # Inform
        print("Iteration {} of {}.".format(i, NUMBER_OF_ITERATIONS))
        # Generate evidence and characteristics
        evidence, characteristics = population.generate()
        identity = evidence.pop()
        # Instantiate
        result_class, entropy = c1.instantiate(evidence, identity)
        # Generate labels
        T = generate_label(result_class, entropy, evidence, identity, characteristics[0])
        # Fuse into previous knowledge
        c1.fuse(T)
        # Calculate accuracy
        accuracy.append(calculate_accuracy(user_characteristics, population.get_users()))

    # Report
    #c1.report()

    with open("cenas.txt", "w") as results_file:
        for item in accuracy:
            results_file.write("{}\n".format(item))

    # Show learned user characteristics
    #pprint.pprint(user_characteristics)

    # accuracy, count, correct = zip(*accuracy)

    # plt.subplot(3,1,1)
    # plt.plot(accuracy)
    # plt.title("Accuracy")
    # plt.subplot(3,1,2)
    # plt.plot(count)
    # plt.title("Total Combinations Learned")
    # plt.subplot(3,1,3)
    # plt.plot(correct)
    # plt.title("Correct Classifications")
    # plt.tight_layout()
    # plt.show()

    # Show accuracy
    #pprint.pprint(accuracy)


def plot_from_file(filename):
    """ Tiny function to plot the data files produced by the iterative tests. """
    accuracy = []
    count = []
    correct = []
    with open(filename) as results_file:
        for line in results_file:
            line = line.strip()
            line = line.strip("[] ")
            line = line.split(",")
            accuracy.append(float(line[0]))
            count.append(float(line[1]))
            correct.append(float(line[2]))


    plt.subplot(3,1,1)
    plt.plot(accuracy)
    plt.ylim([0.0, 1.1])
    plt.title("Accuracy")
    plt.subplot(3,1,2)
    plt.plot(count)
    plt.title("Total Combinations Learned")
    plt.subplot(3,1,3)
    plt.plot(correct)
    plt.title("Correct Classifications")
    plt.tight_layout()
    plt.savefig("no_fusion.pdf")


if __name__=="__main__":
    # Configure Matplotlib
    mpl.rcParams['ps.useafm'] = True
    #mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['axes.ymargin'] = 0.1

    # Run tests
    plot_from_file("/home/vsantos/Desktop/user_model/figs/acc_count_15000_2evidence_10space_nofuse.txt")
    #iterative_test()