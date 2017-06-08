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