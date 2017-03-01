#!/usr/bin/python3
""" A few tests used to fine-tune the fusion and learning mechanisms. """
import probt
import numpy as np
import pprint

# Custom import
import plot_user

def test_1_input():
    # Define Variables
    E_1 = probt.plVariable("E_1", probt.plIntegerType(1, 5))
    C_1 = probt.plVariable("C_1", probt.plIntegerType(1, 5))

    # Define/fill in likelihood
    P_E_1_given_C_1 = probt.plDistributionTable(E_1, C_1)
    values = probt.plValues(E_1 ^ C_1)
    for i in range(5):
        values[C_1] = i+1
        #P_E_1_given_C_1.push(probt.plNormal(E_1, i+1, 0.5), values)
        P_E_1_given_C_1.push(probt.plUniform(E_1), values)
        #P_E_1_given_C_1.instantiate(values).plot("graphs/c_{}.gnuplot".format(i+1))
    print("Initial likelihood", P_E_1_given_C_1)

    # Generate prior (don't worry, probt normalizes it for us)
    P_C_1 = probt.plProbTable(C_1, [1]*5)
    print("Prior", P_C_1)

    # Lists for keeping results
    in_sequence = []
    class_sequence = []
    prob_sequence = []
    

    for i in range(30):
        # Generate evidence
        in_val = np.random.randint(1, 6)
        in_sequence.append(in_val)

        # Define joint distribution
        joint = probt.plJointDistribution(E_1 ^ C_1, P_C_1*P_E_1_given_C_1)
        
        # Define resulting conditional distribution
        P_C_1_given_E_1 = joint.ask(C_1, E_1)
        
        # Define evidence
        evidence = probt.plValues(E_1)
        evidence[E_1] = in_val
        
        # Instantiate/Compile into resulting distribution
        result = P_C_1_given_E_1.instantiate(evidence)
        result_class = result.n_best(1)[0].to_int()
        result_prob = result.tabulate()[1][result_class-1]
        class_sequence.append(result_class)
        prob_sequence.append(result_prob)

        # Update likelihood
        values[C_1] = result_class
        # Instantiate P(E_1|C_1=result)
        dist = P_E_1_given_C_1.instantiate(values).tabulate()[1]
        # Reinforce the corresponding evidence
        dist[in_val-1] = 2*dist[in_val-1]
        # Push into the distribution
        P_E_1_given_C_1.push(values, dist)

        # Report on what happened
        #print("Joint Distribution:", joint)
        #print("Evidence:", evidence)
        #print("Question:", P_C_1_given_E_1)
        #print("Result distribution:", result)
        #print("Table:", result.compile())
        #print("Selected class:", result_class)
        #print("Probability:", result_prob)
        #print("New P(E_{}|C_1={}):".format(in_val, result_class))
        #print(dist)
        #result.plot("graphs/cenas.gnuplot")

    # Print final output
    zipped = zip(in_sequence, class_sequence, prob_sequence)
    print("(input, output, probability)")
    pprint.pprint(list(zipped))


def test_2_input():
    # Define Variables
    E_1 = probt.plVariable("E_1", probt.plIntegerType(1, 5))
    E_2 = probt.plVariable("E_2", probt.plIntegerType(1, 5))
    C_1 = probt.plVariable("C_1", probt.plIntegerType(1, 5))

    input_variables = probt.plVariablesConjunction([E_1,E_2])

    # Define/fill in likelihood
    P_E_given_C_1 = probt.plDistributionTable(E_1^E_2, C_1)
    values = probt.plValues(E_1 ^ E_2 ^ C_1)
    for i in range(5):
        values[C_1] = i+1    
        P_E_given_C_1.push(probt.plUniform(input_variables), values)
    print("Initial likelihood", P_E_given_C_1)

    # Generate prior (don't worry, probt normalizes it for us)
    P_C_1 = probt.plProbTable(C_1, [1]*5)
    print("Prior", P_C_1)

    # Lists for keeping results
    in_sequence = []
    class_sequence = []
    prob_sequence = []
    

    for i in range(10):
        # Generate evidence
        #in_val = [np.random.randint(1, 6), np.random.randint(1, 6)]
        in_val=[1,2]
        in_sequence.append(in_val)

        # Define joint distribution
        joint = probt.plJointDistribution(input_variables ^ C_1, P_C_1*P_E_given_C_1)
        
        # Define resulting conditional distribution
        P_C_1_given_E = joint.ask(C_1, input_variables)
        
        # Define evidence
        evidence = probt.plValues(input_variables)
        evidence[input_variables[0]] = in_val[0]
        evidence[input_variables[1]] = in_val[1]
        
        # Instantiate/Compile into resulting distribution
        result = P_C_1_given_E.instantiate(evidence)
        result_class = result.n_best(1)[0].to_int()
        result_prob = result.tabulate()[1][result_class-1]
        class_sequence.append(result_class)
        prob_sequence.append(result_prob)

        # Update likelihood
        values[C_1] = result_class
        # Instantiate P(E_1|C_1=result)
        instantiation = P_E_given_C_1.instantiate(values).tabulate()
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
            if set([True if elem[i] == in_val[i] else False for i in range(elem.size())]) == {True}:
                dist[i] = 2*dist[i]
        # Push into the distribution
        P_E_given_C_1.push(values, dist)

        # Report on what happened
        #print("Joint Distribution:", joint)
        #print("Evidence:", evidence)
        #print("Question:", P_C_1_given_E)
        #print("Result distribution:", result)
        #print("Table:", result.compile())
        #print("Selected class:", result_class)
        #print("Probability:", result_prob)
        #print("New P(E_{}|C_1={}):".format(in_val, result_class))
        #print(dist)
        #result.plot("graphs/cenas.gnuplot")

    # # Print final output
    zipped = zip(in_sequence, class_sequence, prob_sequence)
    print("(input, output, probability)")
    pprint.pprint(list(zipped))
    return class_sequence


if __name__=="__main__":
    #test_1_input()
    C1 = test_2_input()
    C2 = test_2_input()
    C3 = test_2_input()

    plot_user.plot_users(C1, C2, C3)