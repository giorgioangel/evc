# Exploitation vs Caution

This is the official repository of the paper "An Offline Risk-aware Policy Selection Method for Bayesian
Markov Decision Processes".

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
conda env create -f environment.yml
```

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
trivial = data['t']  # performance of policy trivial
lmdp = data['l']  # performance of policy mean bayesian
qlmdp = data['q']  # performance of policy EvC quantile (VaR)
clmdp = data['c']  # performance of policy EvC CVaR
olmdp = data['o']  # performance of policy EvC optimistic
spibb = data['s']  # performance of policyspibb
bopah = data['b']  # performance of policy bopah
bcrg = data['g'] # performance of policy bcrg
norbur = data['f'] # performance of policy norbur
norbusr = data['sr'] # performance of policy norbusr
norbuvr = data['vr'] # performance of policy norbuvr
random_res = data['v'] # performance of policy random_res
unovar = data['uq'] # performance of policy selected by UNO VaR
unocvar = data['uc'] # performance of policy selected by UNO CVaR
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
├── algorithms/
│      ├── ...
│      ├── ( R scripts for different solvers)
│      └── ...
├── lib/
│    └── norbu_lib.R
├── flake64/
├── chain/
├── ring/
├── chain.R
├── chain_ev.R
├── ring.R
├── ring_ev.R
├── flake64.R
├── flake64_ev.R
├── evc.py  # Main File
├── model_utilities.py  # Contains some necessary functions
├── environments.py  # Contains the definitions of the environments
├── solvers.py  # Contains improved MDP solvers with numba
├── random_policy_perf.py  # Computing of random policy performance
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

Code and documentation copyright 2022 the authors. Code released under the [GNU-AGPLv3 License]

The implementation of BOPAH and SPIBB was taken from: https://github.com/KAIST-AILab/BOPAH
The R solvers were taken from: https://github.com/marekpetrik/CRAAM

Enjoy