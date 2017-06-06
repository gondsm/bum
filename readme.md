This repository contains a ROS package for BUM, a Bayesian User Model. Basic testing does not require ROS at all, in fact you can just `cd` into the `scripts` folder and run 

```
source prep_pypl_env.sh

python3 user_model_tests.py
```

to run the basic tests.

This package depends on the ProBT probabilistic computation package, which must be installed or manually added to the PYTHONPATH via a script similar to `prep_pypl_env.sh`.