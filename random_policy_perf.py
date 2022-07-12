#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Giorgio Angelotti#
# This file is part of EvC.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import multiprocessing as mp
import numpy as np
import mdptoolbox
import pandas as pd
import subprocess
from math import ceil
from environments import Ring
from model_utilities import dirichlet_sample, quantile_calc, learn_transition, oracle_mopo, oracle_morel
from solvers import policy_iteration, policy_eval_i, pol_model, policy_eval, policy_eval_i_s
from finite_mdp import MDP, SPIBB, BOPAH, Alpha  # CODE TAKEN FROM: https://github.com/KAIST-AILab/BOPAH

# FUNCTION THAT TAKES A LIST OF POLICIES AND PUTS ALL THE STATES WITH CORRESPOND TO THE SAME ACTION FOR ALL POLICIES
# IN A LIST (good_indices) AND ALL THE OTHERS IN ANOTHER LIST (bad_indices)
# Input: pol: list of policies, state_space: size of the state space
# Output: bad_indices, good_indices: lists
def policy_idx(pol, state_space):
    bad_indices = []
    good_indices = []
    for s in range(state_space):
        for pid in range(len(pol)):
            if pid == 0:
                act = policies[0][s]
            elif pol[pid][s] != act:
                bad_indices.append(s)
                break
            elif pid == len(pol)-1 and pol[-1][s] == act:
                good_indices.append(s)
    return bad_indices, good_indices


# FUNCTION THAT SOLVES THE BAYESIAN MDP FOR 4 POSSIBLE UTILITY FUNCTIONS (expected value, VaR, CVaR, Optimistic)
# IT EXPLOITS THE FUNCTIONS dirichlet_sample AND quantile_calc TO AUTOMATICALLY DECIDE WHEN IT HAS AN ACCURATE
# ENOUGH ESTIMATE OF THE QUANTILES
# Input: size of the action space, size of the state space, immediate reward matrix, gamma, initial state distr.,
#        transition counter N, most likely transition M, quantile to estimate, desired confidence,
#        list of policies to search over, performing extra_search (0 = No, 1 = Yes)
# Output: Best policies wrt 4 possible utilities functions (Expected Value, VaR, CVaR, Optimistic)
def solve_lmdp(action_space, state_space, reward, gamma, init, N,
               M, quantile, confidence, env, policies=None, search=0):
    # if policies is None, then perform the search over the set of all possible policies
    if policies is None:
        from itertools import product
        policies = list(product(range(action_space), repeat=state_space))

    res = []
    for policy in policies:
        res.append(quantile_calc(np.array(policy, dtype=np.int32), dirichlet_sample, N+1, M, reward, gamma, init,
                                 quantile, 5000, 0.01, confidence=confidence, max_size=50000, evaluator=policy_eval_i))
    output = []  # expected value
    output2 = []  # VaR
    output3 = []  # CVaR
    output4 = []  # Optimistic
    for i in range(len(res)):
        output.append(res[i][0])
        output2.append(res[i][2])
        output3.append(res[i][3])
        output4.append(res[i][4])

    if search == 1:  # if the extra search option is active, search for extra policies
        bidx, gidx = policy_idx(policies, state_space)
        probs = np.zeros((len(bidx), action_space))
        for i in range(len(res)):
            for j in range(len(bidx)):
                probs[j, res[i][1][bidx[j]]] += np.exp(res[i][0])
        for j in range(len(bidx)):
            probs[j] /= np.sum(probs[j])

        new_pol = [res[0][0]]
        for k in range(len(policies)):
            policies[k] = policies[k].astype(np.int32)
        for i in range(10):  # generate at max 10 differents policies
            tempol = np.copy(res[0][1]).astype(np.int32)
            check1 = 1  # check whether the policy is in the original list
            check2 = 1  # check whether the policy is in the new list
            stopper = 0  # iteration counter for each trial
            while check1 >= 1 and check2 >= 1 and stopper < 100:
                stopper += 1
                check1 = 0
                check2 = 0
                for j in range(len(bidx)):
                    a = np.random.choice(action_space, p=probs[j])
                    tempol[bidx[j]] = a
                for u in range(len(policies)):
                    if np.all(tempol == policies[u]):
                        check1 += 1
                for u in range(len(new_pol)):
                    if np.all(tempol == new_pol[u]):
                        check2 += 1
            if check1 == 0 and check2 == 0:
                new_pol.append(np.copy(tempol))
        new_pol.pop(0)
        z = len(res)
        if len(new_pol) > 0:
            for policy in new_pol:
                res.append(
                    quantile_calc(np.array(policy, dtype=np.int32), dirichlet_sample, N + 1, M, reward, gamma, init,
                                  quantile, 5000, 0.01, confidence=confidence, max_size=50000, evaluator=policy_eval_i))
            for i in range(z, len(res)):
                output.append(res[i][0])
                output2.append(res[i][2])
                output3.append(res[i][3])
                output4.append(res[i][4])

    index = np.argmax(output)
    index2 = np.argmax(output2)
    index3 = np.argmax(output3)
    index4 = np.argmax(output4)

    return res[index][1], res[index2][1], res[index3][1], res[index4][1]


# FUNCTION WHICH SOLVES AN MOREL FOR A GIVEN KAPPA AND THRESHOLD ALPHA WITH ORACLE GIVEN UNCERTAINTY
# Inputs: kappa, alpha, true transition matrix T, most likely transition matrix M, immediate reward R,
#          initial state distribution ins, optimal performance, starting policy of policy iteration
# Output: performance of Morel
def solve_morel(kappa, alpha, T, M, R, gamma, ins, V_opt, pol, env):
    model_morel, reward_morel = oracle_morel(kappa, alpha, T, M, R)
    init_morel = np.zeros(model_morel.shape[1])
    init_morel[:-1] = ins
    poly = np.zeros(model_morel.shape[1], dtype=np.int32)
    poly[:-1] = pol
    policy_morel = policy_iteration(model_morel, reward_morel, gamma, poly)[:-1]
    output = policy_eval_i(T, R, policy_morel, gamma, ins) / V_opt
    return output


if __name__ == '__main__':
    import argparse

    # numba compiling
    comp = Ring()
    a, b = pol_model(comp.T, comp.R, np.zeros(comp.T.shape[1], dtype=np.int32))
    policy_eval(a, b, 0.1)
    cpol = policy_iteration(comp.T, comp.R, 0.1, np.zeros(comp.T.shape[1], dtype=np.int32))
    policy_eval_i(comp.T, comp.R, cpol, 0.1, comp.init)
    del comp, cpol, a, b
    #print('Compiled')

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', action='store', dest='environment',
                        help='Select the environment type between "ring", "frozen16", "frozen64" and "medical"')

    parser.add_argument('-b', action='store', dest='batches',
                        help='Select the number of different batches for each size')

    parser.add_argument('-m', action='store', dest='tmin',
                        help='Select min number of trajectories in a batch')

    parser.add_argument('-t', action='store', dest='trajectories',
                        help='Select max number of trajectories in a batch')

    parser.add_argument('-s', action='store', dest='steps',
                        help='Select fixed number of steps for each trajectory')

    parser.add_argument('-q', action='store', dest='quantile',
                        help='Select the quantile to estimate')

    parser.add_argument('-c', action='store', dest='confidence',
                        help='Select the confidence for the quantile estimation')

    parser.add_argument('-n', action='store', dest='number',
                        help='Select the simulation number')

    parser.add_argument('-N', action='store', dest='sampled',
                        help='Select the number of sampled Models to solve')

    parser.add_argument('-f', action='store', dest='full',
                        help='Performing full policy search')  # value 0 (False), 1 (True)

    config = parser.parse_args()

    if config.environment == 'ring':
        mdp = Ring()
        gamma = 0.90

    elif config.environment == 'chain':
        from environments import Chain
        mdp = Chain()
        gamma = 0.90

    elif config.environment == 'frozen64':
        from environments import FrozenLake
        mdp = FrozenLake(type='random', idd=str(config.number))
        gamma = 0.90

    mdp.to_petrik()
    # Obtaining the optimal policy using MDP Toolbox and the matrix representation (eval_type=0)
    pi_optimal = mdptoolbox.mdp.PolicyIteration(mdp.T, mdp.R, gamma, eval_type=0, max_iter=1e7)
    pi_optimal.run()
    optimal_policy = np.array(pi_optimal.policy).astype(np.int32)

    # Changing the obtained optimal policy using my Policy Iteration
    optimal_policy = policy_iteration(mdp.T, mdp.R, gamma, optimal_policy)
    #print('Optimal policy = '+str(optimal_policy))

    t, r = pol_model(mdp.T, mdp.R, optimal_policy)  # computing the Markov Chain Transition function t, exp. reward r
    V_opt = policy_eval_i(mdp.T, mdp.R, optimal_policy, gamma, mdp.init)  # computing the performance
    #V_opt_quant = iterative_policy_eval_i_quantile(mdp.T, mdp.R, optimal_policy, gamma, mdp.init)
    print(V_opt)
    #print(V_opt_quant)
    Vmax = np.max(policy_eval(t, r, gamma))  # computing Vmax to insert in the Oracle MOPO


    # baseline policy uniform random
    pi_b = np.ones((mdp.states, mdp.actions)) / mdp.actions

    V_rand = policy_eval_i_s(mdp.T, mdp.R, pi_b, gamma, mdp.init)  # computing the performance

    print(V_rand/V_opt)

