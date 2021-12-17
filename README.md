# Exploitation vs Caution

This is the official repository of the paper "Exploitation vs Caution: Bayesian Risk-sensitive Policies for Offline Learning".

## Table of contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Data Analysis](#data-analysis)
- [What's included](#whats-included)
- [Creators](#creators)
- [Copyright and license](#copyright-and-license)


## Requirements
The code has been tested on the minimal version reported.

```shell script
pip install --user -r requirements.txt
```

1. python >= 3.7
2. numpy >= 1.19.0
3. numba >= 0.49.0
4. gym >= 0.17.1
5. scipy >= 1.4.1
6. pymdptoolbox >= 4.0b3



## Quick start
```shell script
python -W ignore evc.py -n id -e environment -b batch -m tmin -t tmax -s steps -q quantile -c confidence -f full -N nmodels
```
Example:
```shell script
python -W ignore evc.py -n 0 -e ring -b 50 -m 4 -t 15 -s 30 -q 0.25 -c 0.01 -f 0 -N 3
```
Output:
```shell script
ring_final_0.npz
```

### Data Analysis:
```python
import numpy as np
data = np.load('ring_final_0.npz')
trivial = data['t']  # trivial
lmdp = data['l']  # mean bayesian
qlmdp = data['q']  # quantile
clmdp = data['c']  # pessimistic
olmdp = data['o']  # optimistic
morel = data['r']  # morel
mopo = data['m']  # mopo
spibb = data['s']  # spibb
bopah = data['b']  # bopah
```

## Parameters
- -n id: integer - id of the simulation
- -e environment: string - ring, frozen64, chain
- -b batch: integer - number of different batches of fixed size
- -m tmin: integer - starting number of different trajectories in a batch
- -t tmax: integer - tmax+tmin is the final number of trajectories in a batch
- -s steps: integer - number of steps in each trajectory
- -q quantile: float - quantile to estimate to compute the VaR-q, CVaR-q, Optimistic-q
- -c confidence: float - confidence interval to estimate the quantile
- -f full: 0/1 - perform the search over the full policy space (0 = False, 1 = True) (Monte Carlo Confident Policy Search)
- -N nmodels: integer - integer sample from the posterior N different models and solve them with a running gamma

## What's included

```text
evc/
└── evc.py  # Main File
├── model_utilities.py  # Contains some necessary functions
├── environments.py  # Contains the definitions of the environments
├── solvers.py  # Contains improved MDP solvers with numba
├── finite_mdp.py  # Contains the solvers for SPIBB and BOPAH (source: https://github.com/KAIST-AILab/BOPAH )
├── requirements.txt  # Contains the requirements
├── README.md  # This file
├── LICENSE # GNU Affero General Public License
├── environment_temp_id.npz  # Temporary output file
└── environment_final_id.npz  # Final output file

```

## Creators

**Creator 1**

Giorgio Angelotti


## Copyright and license

Code and documentation copyright 2021 the authors. Code released under the [GNU-AGPLv3 License]

Enjoy