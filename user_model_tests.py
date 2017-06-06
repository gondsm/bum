#!/usr/bin/python3
""" This script implements the testbench used for producing the results for 
the paper.

T vector = [L (chosen label), E (evidence), h (entropy)]

Commands for creating the videos:
avconv -framerate 10 -f image2 -i zz_clusters_%05d.png -c:v h264 -crf 1 out.mp4
avconv -framerate 10 -f image2 -i zz_pop_%05d.png -c:v h264 -crf 1 out.mp4
"""

# Futures
# This was tested on Python3, but might also work on 2.7.
# The following is DEFINITELY needed:
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

# Constants
# These are global constant variables that guide the instantiation of the
# various objects.
# How many users we'll simulate:
NUMBER_OF_USERS = 60
# How many different values are in each evidence variable:
# (implicitly defines the name and number of evidence variables)
EVIDENCE_STRUCTURE = [3, 3]
# How many different values are in the output variables:
# (implicitly defined the name and number of output variables)
CHARACTERISTICS_STRUCTURE = [10, 10, 10]
# How many iterations the system will run for:
NUMBER_OF_ITERATIONS = 1000

# This *dictionary* will maintain the characteristics of the users.
# It is indexed by the tuple (tuple(evidence) + tuple(identity)), and
# as such maintains the latest label for each combination of evidence
# and identity. No initialization will be done; uninitialized values
# will raise a KeyError.
user_characteristics = dict()

# This dictionary maintains a list of previously-trained combinations,
# to guide the label generator
visited_combinations = dict()


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
        a = [range(1, elem+1) for elem in self._evidence_structure]

        # Initialize characteristics for all evidence combination
        for i in range(self._number_of_users):
            iterator = itertools.product(*a)
            for comb in iterator:
                self._users[comb + (i+1,)] = [np.random.randint(1, elem+1) for elem in self._characteristics_structure]


    def __generate_from_profiles(self):
        """ Generates a population from a set of given profiles. """
        #print("Generating guys from profiles", profiles)
        # Build the ranges for all evidence and create an iterator
        a = [range(1, elem+1) for elem in self._evidence_structure]

        # Initialize characteristics for all evidence combination uniformly
        for i in range(self._number_of_users):
            iterator = itertools.product(*a)
            for comb in iterator:
                self._users[comb + (i+1,)] = [np.random.randint(1, elem+1) for elem in self._characteristics_structure]
        
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
                #print(charac)

                # Put into user population
                self._users[key + (i+1,)] = charac


    def generate(self, user=None):
        """ Generates a combination of evidence and labels, according to
        the profiles loaded in self._users.
        If a user ID is received, it is used to retrieve evidence from that user.
        """
        # Generate evidence
        evidence = [np.random.randint(1, elem+1) for elem in self._evidence_structure]
        if user is not None:
            evidence.append(user)
        else:
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


def generate_label(soft_label, entropy, evidence, identity, hard_label=None, combinations=visited_combinations):
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
        combinations[tuple(T[1] + [T[2]])]
    except KeyError:
        # Define T vector with hard evidence, if possible
        if hard_label is not None:
            T = [hard_label, evidence, identity, 0.001]

    # Return the vector for fusion
    return T


def calculate_accuracy(learned_dict, reference_dict):
    """ This function calculates the classification accuracy given the learned
    characteristics of the user.
    """
    # Initialize counters
    count = 0
    correct = 0

    # Determine the number of visited combinations
    count = len(visited_combinations.items())

    # Go through all combinations
    for key, val in learned_dict.items():
        # Compare results for each characteristic
        for i in range(len(val)):
            if val[i] == reference_dict[key][i]:
                correct += 1/len(val)

    # Compute accuracy
    accuracy = (correct/len(learned_dict.items()))

    # And return a new list
    return [accuracy, count, correct]


def calculate_estimation_error(learned_dict, reference_dict, combinations=visited_combinations, evidence=None):
    """ This function calculates the estimation error given the learned
    characteristics of the user.

    If evidence is received, error is calculated only for that evidence
    """
    # Initialize error
    error = 0
    correct = 0
    count = 0

    # Determine the number of visited combinations
    count = len(combinations.items())

    # Go through all combinations
    for key, val in learned_dict.items():
        # Compare results for each characteristic
        if evidence is None:
            for i in range(len(val)):
                if val[i] == reference_dict[key][i]:
                    correct += 1/len(val)
                else:
                    error += abs(val[i] - reference_dict[key][i])
        elif key[0:-1] == tuple(evidence):
            for i in range(len(val)):
                # If we are in a desired tuple
                error += abs(val[i] - reference_dict[key][i])


    # And return a new list
    return [error, count, correct]


def cluster_population(population, evidence, return_gmm=False, num_clusters=2):
    """ Clusters the given population for the given evidence. 
    Returns the raw Gaussian Mixture if requested.
    """
    # Retrieve users
    user_vectors = []
    for i in range(NUMBER_OF_USERS):
        vec = population[tuple(evidence + [i+1])]
        while len(vec) < 3:
            vec.append(1)
        user_vectors.append(vec)

    # Cluster
    gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type='full').fit(user_vectors)

    # Return some results
    if return_gmm:
        # Return the raw mixture model
        return gmm.means_, gmm.covariances_, gmm
    else:
        # Return the means and covariances of the clusters
        return gmm.means_, gmm.covariances_


def cluster_kl(gmm_p, gmm_q, n_samples=10**5):
    """ Kullback-Leibler Divergence of Gaussian Mixture Models based on a
    Monte-Carlo technique.

    Inspired by https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms
    """
    # Get some samples
    X, _ = gmm_p.sample(n_samples)

    # Get probabilities
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)

    # Calculate Dkl
    return log_p_X.mean() - log_q_X.mean()


def plot_population(population, evidence, filename=None, title=None):
    """ This function plots a population of users, in the dictionary form
    defined globally. If a filename is given, the plot is saved on that
    file.
    """
    # Define a range of colors for the users
    colors=["blue", "red", "orange"]

    # Retrieve users
    user_vectors = []
    for i in range(NUMBER_OF_USERS):
        vec = population[tuple(evidence + [i+1])]
        while len(vec) < 3:
            vec.append(1)
        user_vectors.append(vec)

    # Initialize  and adjust figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_zlim([0,10])
    ax.set_xlabel("$C_1$ [classes]")
    ax.set_ylabel("$C_2$ [classes]")
    ax.set_zlabel("$C_3$ [classes]")
    ax.view_init(elev=15., azim=45)
    if title is not None:
        plt.title(title)

    # Plot/Save
    plt.hold(True)
    for i, user in enumerate(user_vectors):
        # Point
        ax.plot([user[0]], [user[1]], [user[2]], 'o', label='User {}'.format(i+1))
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()


def plot_population_cluster(means, covariances, filename=None, title=None):
    """ Given the result of the E-M algorithm, this function plots the clusters
    that were determined. If a filename is given, the plot is saved on that
    file.
    """
    colors = [[1, 0, 0, 0.2],
              [0, 1, 0, 0.2],
              [0, 0, 1, 0.2],
              [1, 0, 1, 0.2]]
    color_iter = itertools.cycle(colors)

    def plot_ellipsoid(radii, center, axes):
        """ Plots an ellipsoid in the given axes
        radii = [x,y,z]
        center = [x,y,z]
        """
        # Unpack radii
        rx, ry, rz = radii

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + center[2]

        # Plot:
        axes.plot_surface(x, y, z,  rstride=4, cstride=4, linewidth=0, color=next(color_iter))
        #ax.set_xlabel("C_1")
        #ax.set_ylabel("C_2")
        #ax.set_zlabel("C_3")

    # Create figure
    fig = plt.figure()  # Square figure
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=45)

    # Plot clusters
    for mean, cov in zip(means, covariances):
        # Determine axis radii
        v, w = linalg.eigh(cov)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # Plot
        plot_ellipsoid(v, mean, ax)

    # Set plot limits and labels
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    ax.set_xlabel(r"$C_1$ [classes]")
    ax.set_ylabel(r"$C_2$ [classes]")
    ax.set_zlabel(r"$C_3$ [classes]")
    if title is not None:
        plt.title(title)

    # Show/save plot
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()


def plot_from_file(filename, filename2=None, is_pickle=True, out_file="zz_results.pdf", only_error=False):
    """ Tiny function to plot the data files produced by the iterative tests. 

    If two filenames are given, the measurements of both are superimposed.
    The ability to read from non-pickle files is kept to maintain the ability
    to plot older data, obtained before implementing pickle support.
    """
    def plot_kl(iters, kl):
        """ Plots just the K-L Divergence """
        plt.figure(figsize=[12,2])
        plt.ylabel("K-L Divergence")
        plt.xlabel("Iterations (k)")
        plt.plot(iters, kl)
        plt.tight_layout()
        plt.savefig("zz_kl.pdf")

    # Define vectors
    accuracy = []
    accuracy2 = []
    count = []
    correct = []
    correct2 = []
    kl = []
    kl2 = []
    error = []
    error2 = []

    # Read from file(s)
    # Basically, we read the data in the format that it is given to us:
    # [ [accuracy, count, correct classifications], [iteration, K-L divergence], [error, count, correct classifications]]
    if is_pickle:
        with open(filename, "rb") as pickle_file:
            temp, kl, temp_err = pickle.load(pickle_file)
            for elem in temp:
                accuracy.append(elem[0])
                count.append(elem[1])
                correct.append(elem[2])
            for elem in temp_err: 
                error.append(elem[0])
        if filename2 is not None:
            with open(filename2, "rb") as pickle_file:
                temp, kl2, temp_err = pickle.load(pickle_file)
                for elem in temp:
                    accuracy2.append(elem[0])
                    correct2.append(elem[2])
                for elem in temp_err:
                    error2.append(elem[0])
    else:
        with open(filename) as results_file:
            for line in results_file:
                line = line.strip()
                line = line.strip("[] ")
                line = line.split(",")
                accuracy.append(float(line[0]))
                count.append(float(line[1]))
                correct.append(float(line[2]))

        if filename2 is not None:
            with open(filename2) as results_file:
                for line in results_file:
                    line = line.strip()
                    line = line.strip("[] ")
                    line = line.split(",")
                    accuracy2.append(float(line[0]))
                    correct2.append(float(line[2]))

    # Plot stuff
    # Determine number of subplots from existence of K-L data
    # num = 3
    # dimensions = (7,8)
    # if kl:
    #     num = 4

    # Actually plot
    # Accuracy
    # plt.figure(figsize=dimensions)
    # plt.subplot(num,1,1)
    # plt.plot([x*100 for x in accuracy], label="fusion")
    # if filename2 is not None:
    #     plt.hold(True)
    #     plt.plot([x*100 for x in accuracy], '--', label="no fusion")
    #     plt.legend(bbox_to_anchor=(1,1.35), loc="upper right", ncol=2)
    # plt.ylim([0.0, 110])
    # plt.xlim([0, len(accuracy)])
    # plt.ylabel("Accuracy (%)")
    # # Combinations
    # plt.subplot(num,1,2)
    # plt.plot(count)
    # plt.xlim([0, len(accuracy)])
    # plt.ylabel("Combinations")
    # # Correct classifications
    # plt.subplot(num,1,3)
    # plt.plot(correct, label="fusion")
    # if filename2 is not None:
    #     plt.hold(True)
    #     plt.plot(correct2, '--', label="no fusion")
    # plt.xlim([0, len(accuracy)])
    # plt.ylabel("Correct\nClassifications")
    # # K-L divergence
    # if kl:
    #     plt.subplot(num,1,4)
    #     plt.plot([k[0] for k in kl], [k[1] for k in kl], label="fusion")
    #     if filename2 is not None:
    #         plt.hold(True)
    #         plt.plot([k[0] for k in kl], [k[1] for k in kl2], '--', label="no fusion")
    #     plt.ylabel("K-L Divergence")
    # # Add iterations label to final plot
    # plt.xlabel("Iterations (k)")

    
    # Plot K-L Divergence
    #plot_kl([k[0] for k in kl], [k[1] for k in kl])

    if only_error:
        # Actually plot
        dimensions = (7,4)
        # Accuracy
        plt.figure(figsize=dimensions)
        plt.plot(error)
        plt.ylim([0.0, 1.1*max(error)])
        plt.xlim([0, len(accuracy)])
        plt.ylabel("Total Error")
    else:
        # Actually plot
        dimensions = (7,5)
        # Accuracy
        plt.figure(figsize=dimensions)
        plt.subplot(2,1,1)
        plt.plot(error)
        plt.ylim([0.0, 1.1*max(error)])
        plt.xlim([0, len(accuracy)])
        plt.ylabel("Total Error")
        # Combinations
        plt.subplot(2,1,2)
        plt.xlim([0, len(accuracy)])
        plt.ylabel("Visited Combinations (n)")
        # Correct classifications
        plt.plot(count, label="Visited combinations")
        #plt.hold(True)
        #plt.plot(correct, '--', label="Correct classifications")
        #plt.xlim([0, len(accuracy)])
        #plt.ylabel("Correct\nClassifications")

        #plt.legend(loc="lower right", ncol=1)

    # Add iterations label to final plot
    plt.xlabel("Iterations (k)")

    # Activate tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(out_file)


def fault_tolerance_test(pickle_file="results_fault.pickle"):
    """ A different scenario where the system's fault-tolerance is tested
    by causing a failure in one of the modules and, afterwards, adding new
    users to the system. """
    # Initialize population simulator
    profiles = dict()
    profiles[(2,3)] = [[2,8,2], [8,2,8], [8,8,4]]
    profile_evidence = [2,3]
    num_clusters = len(profiles[tuple(profile_evidence)])
    population1 = population_simulator(number_of_users=NUMBER_OF_USERS, evidence_structure=EVIDENCE_STRUCTURE, characteristics_structure=CHARACTERISTICS_STRUCTURE, profiles=profiles)
    population2 = population_simulator(number_of_users=NUMBER_OF_USERS, evidence_structure=EVIDENCE_STRUCTURE, characteristics_structure=CHARACTERISTICS_STRUCTURE, profiles=profiles)

    # Initialize counters
    error = []
    accuracy = []
    kl=[]

    # Initialize population models and visited combinations dictionaries
    user_characteristics_local_1 = dict()
    user_characteristics_local_2 = dict()
    visited_combinations_local_1 = dict()
    visited_combinations_local_2 = dict()

    # Define characteristics models
    c_models = []
    for i in range(len(CHARACTERISTICS_STRUCTURE)):
        c_models.append(characteristic_model(EVIDENCE_STRUCTURE, CHARACTERISTICS_STRUCTURE[i], NUMBER_OF_USERS))

    # Reset population back to uniform
    reset_population(user_characteristics_local_1)
    reset_population(user_characteristics_local_2)

    # Run for a few iterations to stabilize clusters
    for i in range(NUMBER_OF_ITERATIONS):
        # Inform on current iteration
        os.system('clear')
        print("Running first iterations until convergence.")
        print("I'm going to save results in {}.".format(pickle_file))
        print("Iteration {} of {}.".format(i+1, NUMBER_OF_ITERATIONS))
        if i > 0:
            print("Previous accuracy: {}.".format(accuracy[-1][0]))
            print("Previous total error: {}".format(error[-1][0]))
        
        # Generate evidence and characteristics
        evidence, characteristics = population1.generate()
        identity = evidence.pop()
        char_result = []

        # Run main cycle for each model
        for j in range(len(c_models)):
            # Instantiate
            result_class, entropy = c_models[j].instantiate(evidence, identity)
            # Generate labels
            T = generate_label(result_class, entropy, evidence, identity, characteristics[j], combinations=visited_combinations_local_1)
            # Append to results vector
            char_result.append(result_class)
            # Fuse into previous knowledge
            c_models[j].fuse(T)

        # Mark combination as visited
        visited_combinations_local_1[tuple(evidence) + (identity,)] = True

        # Update user representation
        user_characteristics_local_1[tuple(evidence) + (identity,)] = copy.copy(char_result)

        # Calculate accuracy
        accuracy.append(calculate_accuracy(user_characteristics_local_1, population1.get_users()))
        error.append(calculate_estimation_error(user_characteristics_local_1, population1.get_users(), visited_combinations_local_1, evidence=profile_evidence))

    # Retrieve clusters from population
    print("Clustering population.")
    clusters = cluster_population(user_characteristics_local_1, profile_evidence, return_gmm=True, num_clusters=num_clusters)

    # Disable one of the modules and infer from clusters for an equal number of iterations
    for i in range(NUMBER_OF_ITERATIONS):
        # Inform on current iteration
        os.system('clear')
        print("Running after first fault.")
        print("I'm going to save results in {}.".format(pickle_file))
        print("Iteration {} of {}.".format(i+1, NUMBER_OF_ITERATIONS))
        if i > 0:
            print("Previous accuracy: {}.".format(accuracy[-1][0]))
            print("Previous total error: {}".format(error[-1][0]))
        
        # Generate evidence and characteristics
        evidence, characteristics = population1.generate()
        identity = evidence.pop()
        char_result = []

        # Run main cycle for each model
        for j in range(len(c_models)):
            # Instantiate
            result_class, entropy = c_models[j].instantiate(evidence, identity)
            # Generate labels
            T = generate_label(result_class, entropy, evidence, identity, characteristics[j], combinations=visited_combinations_local_1)
            # Append to results vector
            char_result.append(result_class)
            # Fuse into previous knowledge
            #c_models[j].fuse(T)

        # Mark combination as visited
        visited_combinations_local_1[tuple(evidence) + (identity,)] = True

        # Update user representation
        user_characteristics_local_1[tuple(evidence) + (identity,)] = copy.copy(char_result)

        # If we're at the right evidence, replace one estimate with one from the clusters
        # (we're simulating a failure in module 0)
        if evidence == profile_evidence:
            # Determine the closest cluster
            min_dist = 1000
            estimate = -1
            for cluster in clusters[0]:
                dist = np.linalg.norm(np.subtract(cluster[1:], evidence[1:]))
                if dist < min_dist:
                    min_dist = dist
                    estimate = np.round(cluster[0])
            # Get from cluster:
            user_characteristics_local_1[tuple(evidence) + (identity,)][0] = estimate

        # Calculate accuracy
        accuracy.append(calculate_accuracy(user_characteristics_local_1, population1.get_users()))
        error.append(calculate_estimation_error(user_characteristics_local_1, population1.get_users(), visited_combinations_local_1, evidence=profile_evidence))


    # Add new users to the pool, model their characteristics and infer the remaining from clusters
    # Define characteristics models
    c_models2 = []
    for i in range(len(CHARACTERISTICS_STRUCTURE)):
        c_models2.append(characteristic_model(EVIDENCE_STRUCTURE, CHARACTERISTICS_STRUCTURE[i], NUMBER_OF_USERS))
    for i in range(NUMBER_OF_ITERATIONS):
        # Inform on current iteration
        os.system('clear')
        print("Running after adding a new population.")
        print("I'm going to save results in {}.".format(pickle_file))
        print("Iteration {} of {}.".format(i+1, NUMBER_OF_ITERATIONS))
        if i > 0:
            print("Previous accuracy: {}.".format(accuracy[-1][0]))
            print("Previous total error: {}".format(error[-1][0]))
        
        # Generate evidence and characteristics
        evidence, characteristics = population2.generate()
        identity = evidence.pop()
        char_result = []

        # Run main cycle for each model
        for j in range(len(c_models2)):
            # Instantiate
            result_class, entropy = c_models2[j].instantiate(evidence, identity)
            # Generate labels
            T = generate_label(result_class, entropy, evidence, identity, characteristics[j], combinations=visited_combinations_local_2)
            # Append to results vector
            char_result.append(result_class)
            # Fuse into previous knowledge
            c_models2[j].fuse(T)

        # Mark combination as visited
        visited_combinations_local_2[tuple(evidence) + (identity,)] = True

        # Update user representation
        user_characteristics_local_2[tuple(evidence) + (identity,)] = copy.copy(char_result)

        # If we're at the right evidence, replace one estimate with one from the clusters
        # (we're simulating a failure in module 0)
        if evidence == profile_evidence:
            # Determine the closest cluster
            min_dist = 1000
            estimate = -1
            for cluster in clusters[0]:
                dist = np.linalg.norm(np.subtract(cluster[1:], evidence[1:]))
                if dist < min_dist:
                    min_dist = dist
                    estimate = np.round(cluster[0])
            # Get from cluster:
            user_characteristics_local_2[tuple(evidence) + (identity,)][0] = estimate

        # Calculate accuracy
        accuracy.append(calculate_accuracy(user_characteristics_local_2, population1.get_users()))
        error1 = calculate_estimation_error(user_characteristics_local_1, population1.get_users(), visited_combinations_local_1, evidence=profile_evidence)
        error2 = calculate_estimation_error(user_characteristics_local_2, population2.get_users(), visited_combinations_local_2, evidence=profile_evidence)
        # Combine errors
        error2[0] = error2[0] + error1[0]
        error.append(error2)


    # Save results to file
    with open(pickle_file, "wb") as pickle_file:
        pickle.dump([accuracy, kl, error], pickle_file)


def iterative_test(pickle_file="results.pickle", clustering=True, plot_clusters=False, fusion=True, epsilon=None, for_video=False):
    """ The "regular" test that is run with this script. 
    
    This function iterates the model for a set number of iterations.

    Inputs:
    pickle_file: the filename to which results are written
    clustering: whether to perform clustering
    plot_clusters: whether to plot the clusters
    fusion: whether to use fusion
    epsilon: if defined, it is used as the convergence coefficient. 
    for_video: if true, plots are printed as png and numbered sequentially
    The system then runs only until the improvement in accuracy drops below
    this value.
    """
    # Initialize population simulator
    profiles = dict()
    profiles[(2,3)] = [[2,8,2], [8,2,8], [8,8,4]]
    #profiles[(2,3)] = [[2,8,2], [8,2,8], [8,8,4]]

    # Initialize population
    # Build the ranges for all evidence and create an iterator
    a = [range(1, elem+1) for elem in EVIDENCE_STRUCTURE]

    # Initialize characteristics for all evidence combination
    # for i in range(NUMBER_OF_USERS):
    #     iterator = itertools.product(*a)
    #     for comb in iterator:
    #         user_characteristics[comb + (i+1,)] = [np.random.randint(1, elem+1) for elem in CHARACTERISTICS_STRUCTURE]
    #         profiles[comb] = [[2,8,2], [8,2,8], [8,8,4]]
    #         pass

    #profile_evidence = [1]*len(EVIDENCE_STRUCTURE)
    profile_evidence = [2,3]
    #profiles[tuple(profile_evidence)] = [[2,2], [3,3]]
    num_clusters = len(profiles[tuple(profile_evidence)])
    population = population_simulator(number_of_users=NUMBER_OF_USERS, evidence_structure=EVIDENCE_STRUCTURE, characteristics_structure=CHARACTERISTICS_STRUCTURE, profiles=profiles)

    # Reset population back to uniform
    reset_population()

    # Define characteristics models
    c_models = []
    for i in range(len(CHARACTERISTICS_STRUCTURE)):
        c_models.append(characteristic_model(EVIDENCE_STRUCTURE, CHARACTERISTICS_STRUCTURE[i], NUMBER_OF_USERS))

    # Lists to maintain the accuracy and K-L divergence (results)
    # Accuracy is actually more than simple accuracy, as defined in 
    # calculate_accuracy()
    accuracy = []
    kl = []
    cluster_record = []
    error = []

    final_iteration = 0

    frame_count = 0

    init = time.time()

    # Run
    for i in range(NUMBER_OF_ITERATIONS):
        # Inform on current iteration
        os.system('clear')
        print("I'm going to save results in {}.".format(pickle_file))
        print("Iteration {} of {}.".format(i+1, NUMBER_OF_ITERATIONS))
        if i > 0:
            print("Previous accuracy: {}.".format(accuracy[-1][0]))
            print("Previous total error: {}".format(error[-1][0]))
        
        # Generate evidence and characteristics
        evidence, characteristics = population.generate()
        identity = evidence.pop()
        char_result = []

        # Run main cycle for each model
        for j in range(len(c_models)):
            # Instantiate
            result_class, entropy = c_models[j].instantiate(evidence, identity)
            # Generate labels
            T = generate_label(result_class, entropy, evidence, identity, characteristics[j])
            # Append to results vector
            char_result.append(result_class)
            # Fuse into previous knowledge
            if fusion == True:
                c_models[j].fuse(T)

        # Mark combination as visited
        visited_combinations[tuple(evidence) + (identity,)] = True

        # Update user representation
        user_characteristics[tuple(evidence) + (identity,)] = copy.copy(char_result)

        # Calculate accuracy
        accuracy.append(calculate_accuracy(user_characteristics, population.get_users()))
        error.append(calculate_estimation_error(user_characteristics, population.get_users()))

        # If i is in a certain value, we calculate the clusters
        if i in range(0, NUMBER_OF_ITERATIONS, 10) and clustering == True:
            # Cluster population
            clusters = cluster_population(user_characteristics, profile_evidence, return_gmm=True, num_clusters=num_clusters)
            #cluster_record.append([clusters[0], clusters[1], i])
            # Calculate divergence to reference population
            kl.append([i, cluster_kl(clusters[2], cluster_population(population.get_users(), profile_evidence, return_gmm=True)[2])])
            if plot_clusters == True:
                # Plot
                if for_video is True:
                    plot_population_cluster(clusters[0], clusters[1], filename="zz_clusters_{0:05d}.png".format(frame_count), title="Iteration {}".format(i))
                    plot_population(user_characteristics, profile_evidence, "zz_pop_{0:05d}.png".format(frame_count), title="Iteration {}".format(i))
                    frame_count += 1
                else:
                    plot_population_cluster(clusters[0], clusters[1], filename="zz_clusters_iter{0:05d}.pdf".format(i))
                    # Plot population
                    plot_population(user_characteristics, profile_evidence, "zz_pop_iter{0:05d}.pdf".format(i))

        # Stop iterating if we reached the accuracy goal
        history_length = 100
        if epsilon is not None and (len(accuracy) > history_length):
            #cenas = np.absolute(sum(np.diff([accuracy[-i][0] for i in range(history_length,0, -1)])))
            cenas = [accuracy[-i][0] for i in range(history_length,0, -1)]
            cenas = max(cenas) - min(cenas)
            if cenas < epsilon:
                final_iteration = i
                break

    # Inform
    os.system("clear")
    print("End of test results:")
    print("The system took {} iterations to converge.".format(final_iteration))
    print("The final accuracy is {}.".format(accuracy[-1][0]))
    print("Execution time was {} seconds.".format(time.time()-init))
    print("Evidence structure is {}.".format(EVIDENCE_STRUCTURE))
    print("Characteristics structure is {}.".format(CHARACTERISTICS_STRUCTURE))
    print("Number of users is {}.".format(NUMBER_OF_USERS))

    # Save results to table
    with open("table.tex", "a") as table_file:
        # $\mathbf{E}$ & $\mathbf{C}$ & $n$ & Conv. Time & Final Acc & Execution Time \\
        table_file.write(", ".join([str(elem) for elem in EVIDENCE_STRUCTURE]))
        table_file.write(" & ")
        table_file.write(", ".join([str(elem) for elem in CHARACTERISTICS_STRUCTURE]))
        table_file.write(" & ")
        table_file.write("{}".format(NUMBER_OF_USERS))
        table_file.write(" & ")
        table_file.write("{}".format(final_iteration))
        table_file.write(" & ")
        table_file.write("{:03.4f}".format(accuracy[-1][0]*100))
        table_file.write(" & ")
        table_file.write("{:03.4f}".format(time.time()-init))
        table_file.write(" \\\\\n")

    # Save results to file
    with open(pickle_file, "wb") as pickle_file:
        pickle.dump([accuracy, kl, error], pickle_file)

    # Plot some clusters
    if plot_clusters == True:
        plot_population(population.get_users(), profile_evidence, "zz_original_pop.pdf")
        plot_population(user_characteristics, profile_evidence, "zz_result_pop.pdf")
        clusters = cluster_population(population.get_users(), profile_evidence, num_clusters=num_clusters)
        plot_population_cluster(*clusters, filename="zz_original_clusters.pdf")
        clusters = cluster_population(user_characteristics, profile_evidence, num_clusters=num_clusters)
        plot_population_cluster(*clusters, filename="zz_result_clusters.pdf")


def reset_population(population=user_characteristics):
    """ This function resets the global population back to uniformity. """
    # Initialize population
    # Build the ranges for all evidence and create an iterator
    a = [range(1, elem+1) for elem in EVIDENCE_STRUCTURE]

    # Initialize characteristics for all evidence combination
    for i in range(NUMBER_OF_USERS):
        iterator = itertools.product(*a)
        for comb in iterator:
            population[comb + (i+1,)] = [np.random.randint(1, elem+1) for elem in CHARACTERISTICS_STRUCTURE]


def debug():
    """ A simple debug function used occasionally to test stuff. """
    # profiles = dict()
    # profiles[(2,3)] = [[2,8,2], [8,2,8], [8,8,4], [2,2,8]]
    # population = population_simulator(NUMBER_OF_USERS, EVIDENCE_STRUCTURE, CHARACTERISTICS_STRUCTURE, profiles=profiles)
    # plot_population(population.get_users(), [2,3])
    # print(cluster_kl(clusters1, clusters2))

    c = characteristic_model([3,3,3], 10, 30)
    likelihood = c.get_likelihood()
    print(likelihood)

    #pass


if __name__=="__main__":
    # Configure Matplotlib
    mpl.rcParams['ps.useafm'] = True
    #mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['text.usetex'] = True
    #mpl.rcParams['text.latex.unicode']=False
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['axes.ymargin'] = 0.1

    # Generate main results figure
    #iterative_test(pickle_file="results_no_fusion.pickle", fusion=False)
    #iterative_test(pickle_file="results_fusion.pickle")
    #plot_from_file("in_paper/results_fusion.pickle", "in_paper/results_no_fusion.pickle", is_pickle=True)

    # Generate the clusters figure
    #iterative_test(clustering=True, plot_clusters=True)

    # Run many tests
    #for i in range(26):
    #    p = Process(target=iterative_test, args=("many_results/results{0:03d}.pickle".format(i),))
    #    p.start()
    #    p.join()
    #plot_several_pickles(["many_results/results{0:03d}.pickle".format(i) for i in range(3)])

    # Run until convergence
    #iterative_test(epsilon=0.005, pickle_file="conv.pickle")
    #plot_from_file("conv.pickle", is_pickle=True)

    # Run the debug function
    #debug()


    # Other stuff
    #iterative_test(pickle_file="results.pickle")
    #plot_from_file("results.pickle", is_pickle=True)
    #fault_tolerance_test()
    plot_from_file("results_fault.pickle", is_pickle=True, only_error=True)