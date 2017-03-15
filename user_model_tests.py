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
NUMBER_OF_USERS = 30
# How many different values are in each evidence variable:
# (implicitly defines the name and number of evidence variables)
EVIDENCE_STRUCTURE = [4, 4] 
# How many different values are in the output variables:
# (implicitly defined the name and number of output variables)
CHARACTERISTICS_STRUCTURE = [10, 10, 10]
# How many iterations the system will run for:
NUMBER_OF_ITERATIONS = 5000

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
        visited_combinations[tuple(T[1] + [T[2]])]
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


def plot_population(population, evidence, filename=None):
    """ This function plots a population of users. """

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
    ax.set_xlabel("C_1")
    ax.set_ylabel("C_2")
    ax.set_zlabel("C_3")
    ax.view_init(elev=15., azim=45)

    # Plot/Save
    plt.hold(True)
    for i, user in enumerate(user_vectors):
        # Point
        ax.plot([user[0]], [user[1]], [user[2]], 'o', label='User {}'.format(i+1))
        
    #ax.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
    #ax.legend(numpoints=1, ncol=3)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_population_cluster(means, covariances, filename=None):
    """ Given the result of the E-M algorithm, this function plots the clusters
    that were determined.
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
        #coefs = (1, 1, 2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
        # Radii corresponding to the coefficients:
        #rx, ry, rz = 1/np.sqrt(coefs)
        rx, ry, rz = radii

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + center[2]

        # Plot:
        axes.plot_surface(x, y, z,  rstride=4, cstride=4, linewidth=0, color=next(color_iter))

        ax.set_xlabel("C_1")
        ax.set_ylabel("C_2")
        ax.set_zlabel("C_3")

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
    ax.set_xlabel("C_1")
    ax.set_ylabel("C_2")
    ax.set_zlabel("C_3")


    # Show plot
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_from_file(filename, filename2=None, is_pickle=False):
    """ Tiny function to plot the data files produced by the iterative tests. 

    If two filenames are given, the accuracies of both are superimposed
    """
    # Allocate vectors
    accuracy = []
    accuracy2 = []
    count = []
    correct = []
    correct2 = []
    kl = []
    kl2 = []

    # Read from file(s)
    if is_pickle:
        with open(filename, "rb") as pickle_file:
            temp, kl = pickle.load(pickle_file)
            for elem in temp:
                accuracy.append(elem[0])
                count.append(elem[1])
                correct.append(elem[2])
        if filename2 is not None:
            with open(filename2, "rb") as pickle_file:
                temp, kl2 = pickle.load(pickle_file)
                for elem in temp:
                    accuracy2.append(elem[0])
                    correct2.append(elem[2])
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
    # Determine number of subplots
    num = 3
    if kl:
        num = 4
    # Actually plot
    plt.subplot(num,1,1)
    plt.plot([x*100 for x in accuracy], label="fusion")
    if filename2 is not None:
        plt.hold(True)
        plt.plot([x*100 for x in accuracy2], label="no fusion")
        plt.legend(loc=5)
    plt.ylim([0.0, 110])
    plt.xlim([0, len(accuracy)])
    plt.ylabel("Accuracy (%)")
    #plt.title("Accuracy")
    plt.subplot(num,1,2)
    plt.plot(count)
    plt.xlim([0, len(accuracy)])
    #plt.title("Total Combinations Learned")
    plt.ylabel("Combinations")
    plt.subplot(num,1,3)
    plt.plot(correct, label="fusion")
    if filename2 is not None:
        plt.hold(True)
        plt.plot(correct2, label="no fusion")
        plt.legend(loc=5)
    plt.xlim([0, len(accuracy)])
    plt.ylabel("Correct Classifications")
    plt.xlabel("Iterations (k)")

    if kl:
        plt.subplot(num,1,4)
        plt.plot([k[0] for k in kl], [k[1] for k in kl])
        plt.title("K-L Divergence")

    plt.tight_layout()
    plt.savefig("zz_results.pdf")


def plot_several_pickles(pickles):
    """ Yummy! """
    accuracy_lists = []
    count_lists = []
    correct_lists = []

    for i, pickle_file in enumerate(pickles):
        with open(pickle_file, "rb") as pickle_file:
            temp, kl = pickle.load(pickle_file)
            if i == 0:
                accuracy_lists = [[] for x in range(len(temp))]
                count_lists = [[] for x in range(len(temp))]
                correct_lists = [[] for x in range(len(temp))]
                #print(accuracy_lists)
            for j, elem in enumerate(temp):
                accuracy_lists[j].append(elem[0])
                count_lists[j].append(elem[1])
                correct_lists[j].append(elem[2])

    # Plot stuff
    # Determine number of subplots
    num = 3
    # Actually plot
    plt.subplot(num,1,1)
    
    ax = plt.gca()
    plt.plot([np.mean(x)*100 for x in accuracy_lists], label="fusion")
    plt.ylim([0.0, 110])
    plt.xlim([0, len(accuracy_lists)])
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy")

    # Add variation rectangle
    plt.hold(True)
    # Maximum and minimum vectors
    maxs = [max(x)*100 for x in accuracy_lists]
    mins = [min(x)*100 for x in accuracy_lists]
    # Determine points (polygons are drawn clockwise)
    points = []
    for i in range(len(maxs)):
        points.append([i, maxs[i]])
    for i in range(len(mins)-1, -1, -1):
        points.append([i, mins[i]])
    # Create patch
    patches = []
    polygon = mpatches.Polygon(points)
    patches.append(polygon)
    # Add patch to axis
    colors = np.linspace(0, 1, len(patches))
    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)

    plt.subplot(num,1,2)
    plt.plot([x[0] for x in count_lists])
    plt.xlim([0, len(accuracy_lists)])
    #plt.title("Total Combinations Learned")
    plt.ylabel("Combinations")
    plt.subplot(num,1,3)
    plt.plot([x[0] for x in correct_lists])
    plt.xlim([0, len(accuracy_lists)])
    plt.ylabel("Correct Classifications")
    plt.xlabel("Iterations (k)")

    plt.tight_layout()
    plt.savefig("zz_multiple_results.pdf")


def iterative_test(pickle_file="results.pickle", clustering=True, plot_clusters=False):
    """ The "regular" test that is run with this script. 
    
    TODO: What does it do?
    """
    # Initialize population simulator
    profiles = dict()
    profiles[(2,3)] = [[2,8,2], [8,2,8], [8,8,4], [2,2,8]]
    num_clusters = len(profiles[(2,3)])
    population = population_simulator(NUMBER_OF_USERS, EVIDENCE_STRUCTURE, CHARACTERISTICS_STRUCTURE, profiles=profiles)

    # Reset population
    reset_population()

    # Define models
    c_models = []
    for i in range(len(CHARACTERISTICS_STRUCTURE)):
        c_models.append(characteristic_model(EVIDENCE_STRUCTURE, CHARACTERISTICS_STRUCTURE[i], NUMBER_OF_USERS))

    # List to maintain the accuracy of the system
    accuracy = []

    # List to maintain the K-L divergence to the original clusters
    kl = []

    # Run
    for i in range(NUMBER_OF_ITERATIONS):
        # Inform
        os.system('clear')
        print("I'm going to save results in {}.".format(pickle_file))
        print("Iteration {} of {}.".format(i+1, NUMBER_OF_ITERATIONS))
        # Generate evidence and characteristics
        evidence, characteristics = population.generate()
        identity = evidence.pop()
        char_result = []
        for j in range(len(c_models)):
            # Instantiate
            result_class, entropy = c_models[j].instantiate(evidence, identity)
            # Generate labels
            T = generate_label(result_class, entropy, evidence, identity, characteristics[j])
            #T = generate_label(characteristics[j], entropy, evidence, identity, characteristics[j])
            # Append to results vector
            char_result.append(result_class)
            # Fuse into previous knowledge
            c_models[j].fuse(T)
        # Mark combination as visited
        visited_combinations[tuple(evidence) + (identity,)] = True
        # Update user representation
        user_characteristics[tuple(evidence) + (identity,)] = copy.copy(char_result)
        # Calculate accuracy
        accuracy.append(calculate_accuracy(user_characteristics, population.get_users()))

        # If i is in a certain value, we calculate the clusters
        if i in range(0, NUMBER_OF_ITERATIONS, int(NUMBER_OF_ITERATIONS/20)) and clustering == True:
            # Cluster population
            clusters = cluster_population(user_characteristics, [2,3], return_gmm=True, num_clusters=num_clusters)
            # Calculate divergence to reference population
            kl.append([i, cluster_kl(clusters[2], cluster_population(population.get_users(), [2,3], return_gmm=True)[2])])
            if plot_clusters == True:
                # Plot
                plot_population_cluster(clusters[0], clusters[1], filename="zz_clusters_iter{0:05d}.pdf".format(i))
                # Plot population
                plot_population(user_characteristics, [2,3], "zz_pop_iter{0:05d}.pdf".format(i))

    # Save results to file (raw text and pickle)
    with open("cenas.txt", "w") as results_file:
        for item in accuracy:
            results_file.write("{}\n".format(item))

    with open(pickle_file, "wb") as pickle_file:
        pickle.dump([accuracy, kl], pickle_file)

    # Plot some clusters
    if plot_clusters == True:
        plot_population(population.get_users(), [2,3], "zz_original_pop.pdf")
        plot_population(user_characteristics, [2,3], "zz_result_pop.pdf")
        clusters = cluster_population(population.get_users(), [2,3], num_clusters=num_clusters)
        plot_population_cluster(*clusters, filename="zz_original_clusters.pdf")
        clusters = cluster_population(user_characteristics, [2,3], num_clusters=num_clusters)
        plot_population_cluster(*clusters, filename="zz_result_clusters.pdf")


def reset_population():
    # Initialize population
    # Build the ranges for all evidence and create an iterator
    a = [range(1, elem+1) for elem in EVIDENCE_STRUCTURE]

    # Initialize characteristics for all evidence combination
    for i in range(NUMBER_OF_USERS):
        iterator = itertools.product(*a)
        for comb in iterator:
            user_characteristics[comb + (i+1,)] = [np.random.randint(1, elem+1) for elem in CHARACTERISTICS_STRUCTURE]


def debug():
    profiles = dict()
    profiles[(2,3)] = [[2,8,2], [8,2,8], [8,8,4], [2,2,8]]
    population = population_simulator(NUMBER_OF_USERS, EVIDENCE_STRUCTURE, CHARACTERISTICS_STRUCTURE, profiles=profiles)
    plot_population(population.get_users(), [2,3])
    #print(cluster_kl(clusters1, clusters2))


if __name__=="__main__":
    # Configure Matplotlib
    mpl.rcParams['ps.useafm'] = True
    #mpl.rcParams['pdf.use14corefonts'] = True
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['axes.ymargin'] = 0.1

    #
    #reset_population()
    


    # Run tests
    # for i in range(26):
    #     p = Process(target=iterative_test, args=("many_results/results{0:03d}.pickle".format(i),))
    #     p.start()
    #     p.join()
    plot_several_pickles(["many_results/results{0:03d}.pickle".format(i) for i in range(3)])
    #iterative_test()
    #plot_from_file("cenas.txt")
    #plot_from_file("results.pickle", is_pickle=True)
    #plot_from_file("many_results/results000.pickle", is_pickle=True)
    #debug()

    # Generate one of the paper figures
    #plot_from_file("/home/vsantos/Desktop/user_model/figs/acc_count_15000_2evidence_10space.txt", "/home/vsantos/Desktop/user_model/figs/acc_count_15000_2evidence_10space_nofuse.txt")