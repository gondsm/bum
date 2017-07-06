""" The bum_utils script
This script contains utilities, mainly used by the user_model_tests scripts to
operate normally.
"""

def generate_label(soft_label, entropy, evidence, identity, hard_label=None, combinations=None):
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


# From here on out, the functions adhere to the new formulation of "population"
# and are to be used by the new ROS nodes.
def reset_population(population, gcd, user_ids=None):
    """ Resets user population back to a uniform distribution. 

    This is done according to the GCD and the population generated is of the
    form population["characteristic_id"] = {[evidence, identity]: characteristic_value}
    meaning that there is a main entry for each characteristic, and within it
    the values are contained in a dict for every combination of evidence and
    identity (indexed as a single tuple).

    If a vector of user_ids is received, the population is only reset for those IDs.
    """
    # Population is split by characteristics
    for char in gcd["C"]:
        # If a characteristic is not active, we do not touch it
        if char not in gcd["Config"]["Active"]:
            continue
        # Create new dictionary for this characteristic
        population[char] = dict()

        # Create evidence structure for this variable
        evidence_structure = []
        for key in gcd["C"][char]["input"]:
            evidence_structure.append(gcd["E"][key]["nclasses"])

        # Initialize characteristics for all evidence combination
        a = [range(0, elem) for elem in evidence_structure]
        if user_ids is not None:
            ids = user_ids
        else:
            ids = range(gcd["nusers"])
        for i in ids:
            iterator = itertools.product(*a)
            for comb in iterator:
                population[char][comb + (i,)] = np.random.randint(0, gcd["C"][char]["nclasses"])


def cluster_population(population, gcd, return_gmm=False, num_clusters=2):
    """ Clusters the given population, assuming all characteristics are user-bound. 
    Returns the raw Gaussian Mixture object if requested.
    """
    # Retrieve users, each user is an element of the vector
    user_vectors = [[] for i in range(gcd["nusers"])]
    for char in gcd["Config"]["Active"]:
        for ev_comb in population[char]:
            user = ev_comb[-1]
            user_vectors[user].append(population[char][(user,)])
    # Clear out empty vectors
    user_vectors = [vec for vec in user_vectors if vec]

    # Cluster
    gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type='full').fit(user_vectors)

    # Return some results
    if return_gmm:
        # Return the raw mixture model
        return gmm.means_, gmm.covariances_, gmm
    else:
        # Return the means and covariances of the clusters
        return gmm.means_, gmm.covariances_