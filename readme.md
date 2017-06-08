# Introduction

This repository contains a ROS package for BUM, a Bayesian User Model. 

# Global Characteristic Description
The GDC file describes the problem we're tackling. It specifies the characteristics to be estimated, the evidence that each characteristic takes, as well as the ranges for all of these variables. It also specifies which characteristics are active, so that the same GDC file can be used by several nodes with minimal changes. Lastly, it specifies how many users the system is expected to encounter.

An example of a GDC file is as follows (included in the `config` folder):

```
# Characteristics
C:
  C1: 
    input: 
      - E1 
      - E2 
      - E3
    nclasses: 10
  C2: 
    input: 
      - E1 
      - E2 
    nclasses: 10
  C3:
    input:
      - E3
    nclasses: 10

# Evidence
E:
  E1:
    nclasses: 10
  E2:
    nclasses: 10
  E3:
    nclasses: 10

# Specify characteristics this node will word on
Active:
  - C1
  - C2
  - C3

# Specify the number of users
nusers: 10
```

# Testing

Basic testing does not require ROS at all, in fact you can just `cd` into the `scripts` folder and run 

```
source prep_pypl_env.sh

python3 user_model_tests.py
```

to run the basic tests.

This package depends on the ProBT probabilistic computation package, which must be installed or manually added to the PYTHONPATH via a script similar to `prep_pypl_env.sh`.

Useful command to publish evidence:

```
rostopic pub /bum/evidence bum_ros/Evidence "{values:[2,3,2], evidence_ids:['E1', 'E2', 'E3'], user_id: 1}"
```

Useful command to publish a tuple:

```
rostopic pub /bum/tuple bum_ros/Tuple "{char_id: 'C1', characteristic: 4, evidence: [2, 3, 2], user_id: 1, h: 0.8}"
```


