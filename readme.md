# Introduction

This repository contains the ROS implementation of BUM, a Bayesian User Model. It is divided in two packages:

* `bum_ros` contains the core engine, messages, auxiliary scripts, etc... that make up the BUM system. It can be used separately and only depends on ROS and the ProBT probability computation package.
* `bum_ros_conductor` contains the scripts needed for interaction with the user, namely for data gathering. It depends heavily on the `conductor` ROS package developed for the GrowMeUp project. However, it is completely decoupled from the `bum_ros` package, meaning that it can easily be replaced with another data source, or with a module that generates data from a different robot.


# Global Characteristic Description
The GDC file describes the problem we're tackling. It specifies the characteristics to be estimated, the evidence that each characteristic takes, as well as the ranges for all of these variables. It also specifies which characteristics are active, so that the same GDC file can be used by several nodes with minimal changes. Lastly, it specifies how many users the system is expected to encounter.

An example of a GDC file can be found in the `config` folder.

# Testing Without ROS

Basic testing does not require ROS at all, in fact you can just `cd` into the `bum_ros/scripts` folder and run 

```
source prep_pypl_env.sh

python3 user_model_tests.py
```

to run the basic tests.

The ProBT probabilistic computation package must be installed or manually added to the PYTHONPATH via a script similar to `prep_pypl_env.sh`.

# Testing with ROS

## Simulation

For testing with ROS, the basic system can be started by running

```
rosrun bum_ros bum_ros_node.py

rosrun bum_ros data_manager.py

rosrun bum_ros evaluator.py
```

This starts the main nodes with the default GCD, and the system is ready to receive input.

## With interaction

Once the main nodes are running, running with interaction on the GrowMeUp system is a matter of running

```
rosrun bum_ros_conductor bum_conductor.py
```

this starts the interaction node, and the GrowMu robot should start speaking.

## Testing different topologies

*TODO*

## Useful commands

Useful command to publish evidence:

```
rostopic pub /bum/evidence bum_ros/Evidence "{values:[2,3,2], evidence_ids:['E1', 'E2', 'E3'], user_id: 1}"
```

Useful command to publish a tuple:

```
rostopic pub /bum/tuple bum_ros/Tuple "{char_id: 'C1', characteristic: 4, evidence: [2, 3, 2], user_id: 1, h: 0.8, hard: False}"
```
