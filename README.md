# SPOD

This repository is the implementation of the RL algorithm: Soft policy optimization using dual-track advantage estimator (SPOD).

## Prerequisites

SPOD require Python3  (>=3.5) with the development header. The detailed install steps of the required software are shown in  https://github.com/openai/baselines

## Installation

* Clone the repo and cd into it:

  ```
  git clone https://github.com/Code-Papers/SPOD
  cd SPOD
  ```

* Install SPOD

  ```
  pip install -e .
  ```

## Training models

The mode to train your agent is:

```
python -m baselines.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```

**Example: SPOD with MuJoCo Humanoid**

```
python -m baselines.run --alg=tdppo --env=Humanoid-v2 --network=mlp --num_timesteps=1e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy
```

