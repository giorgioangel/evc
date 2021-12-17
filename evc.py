#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 Giorgio Angelotti
#
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
                                 quantile, 5000, 0.01, confidence=confidence, max_size=50000))
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
                                  quantile, 5000, 0.01, confidence=confidence, max_size=50000))
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
        mdp = FrozenLake(type='random')
        gamma = 0.90

    # Obtaining the optimal policy using MDP Toolbox and the matrix representation (eval_type=0)
    pi_optimal = mdptoolbox.mdp.PolicyIteration(mdp.T, mdp.R, gamma, eval_type=0, max_iter=1e7)
    pi_optimal.run()
    optimal_policy = np.array(pi_optimal.policy).astype(np.int32)

    # Changing the obtained optimal policy using my Policy Iteration
    optimal_policy = policy_iteration(mdp.T, mdp.R, gamma, optimal_policy)
    #print('Optimal policy = '+str(optimal_policy))

    t, r = pol_model(mdp.T, mdp.R, optimal_policy)  # computing the Markov Chain Transition function t, exp. reward r
    V_opt = policy_eval_i(mdp.T, mdp.R, optimal_policy, gamma, mdp.init)  # computing the performance
    Vmax = np.max(policy_eval(t, r, gamma))  # computing Vmax to insert in the Oracle MOPO

    # initializing arrays for the results
    trivial = np.zeros((int(config.trajectories)-int(config.tmin), int(config.batches)))
    lmdp = np.zeros_like(trivial)
    qlmdp = np.zeros_like(trivial)
    clmdp = np.zeros_like(trivial)
    olmdp = np.zeros_like(trivial)
    mopo = np.zeros((trivial.shape[0], trivial.shape[1], 5))
    morel = np.zeros((trivial.shape[0], trivial.shape[1], 5))
    spibb0 = np.zeros((trivial.shape[0], trivial.shape[1], 7))
    bopah0 = np.zeros((trivial.shape[0], trivial.shape[1], 1, 1))

    # baseline policy uniform random
    pi_b = np.ones((mdp.states, mdp.actions)) / mdp.actions

    for traj_number in range(trivial.shape[0]):
        for trial in range(trivial.shape[1]):
            #print(str(traj_number)+'_'+str(trial), end='\n')
            # generate traj_number+tmin trajectories of config.steps each
            hist, init, trajectories, batch_traj = mdp.generate_trajectories(traj_number + int(config.tmin), int(config.steps))
            M, N = learn_transition(mdp.T, hist)  # M = most likely T, N = transition counter
            trivial_policy = policy_iteration(M, mdp.R, gamma, optimal_policy)  # solving the most likely
            #print('Trivial policy = ' + str(trivial_policy))
            # performance of trivial_policy in the real model, normalized by the one of the optimal policy
            trivial[traj_number, trial] = policy_eval_i(mdp.T, mdp.R, trivial_policy, gamma, mdp.init) / V_opt

            models = dirichlet_sample(N+1, int(config.sampled))  # sampling config.sampled models from the posterior
            temp = np.zeros((1, M.shape[0], M.shape[1], M.shape[2]))
            temp[0] = M
            models = np.vstack((models, temp))  # add M to the sampled models

            # Find the optimal policy for the MDP defined using the transitions in Models and a running gamma
            # that goes from 0.2 to the evaluation gamma with steps of 0.2
            pool = mp.Pool(mp.cpu_count())  # multiprocessing
            inputs = [(models[j], mdp.R, i * 0.2, trivial_policy) for i in range(1, ceil(gamma / 0.2))
                      for j in range(models.shape[0])]
            for j in range(models.shape[0]-1):  # -1 because I already solved the trivial model
                inputs.append((models[j], mdp.R, gamma, trivial_policy))
            polput = pool.starmap_async(policy_iteration, inputs)
            policies = polput.get()
            pool.close()
            pool.join()

            policies.append(trivial_policy)
            # convert the list of numpy vectors in list of tuples
            for idx in range(len(policies)):
                policies[idx] = tuple(policies[idx])
            policies = list(set(policies))  # remove duplicates
            # reconvert the tuples to numpy arrays
            for idx in range(len(policies)):
                policies[idx] = np.array(policies[idx], dtype=np.int32)

            if len(policies) > 1 or int(config.full) == 1:  # if there is more than one policy in the list
                # Solve the Bayesian MDP (Expected Value, VaR, CVaR, Optimistic)
                if int(config.full) == 1:
                    pol_best, pol_q_best, pol_c_best, pol_o_best = solve_lmdp(mdp.actions, mdp.states, mdp.R, gamma,
                                                                              init, N, M, float(config.quantile),
                                                                              float(config.confidence),
                                                                              str(config.environment), policies=None)

                else:
                    pol_best, pol_q_best, pol_c_best, pol_o_best = solve_lmdp(mdp.actions, mdp.states, mdp.R, gamma,
                                                                              init, N, M, float(config.quantile),
                                                                              float(config.confidence),
                                                                              str(config.environment),
                                                                              policies=policies)
                lmdp[traj_number, trial] = policy_eval_i(mdp.T, mdp.R, pol_best, gamma, mdp.init) / V_opt
                qlmdp[traj_number, trial] = policy_eval_i(mdp.T, mdp.R, pol_q_best, gamma, mdp.init) / V_opt
                clmdp[traj_number, trial] = policy_eval_i(mdp.T, mdp.R, pol_c_best, gamma, mdp.init) / V_opt
                olmdp[traj_number, trial] = policy_eval_i(mdp.T, mdp.R, pol_o_best, gamma, mdp.init) / V_opt
            else:  # the only policy is the one of the most likely model, the performance is the same
                lmdp[traj_number, trial] = trivial[traj_number, trial]
                qlmdp[traj_number, trial] = trivial[traj_number, trial]
                clmdp[traj_number, trial] = trivial[traj_number, trial]
                olmdp[traj_number, trial] = trivial[traj_number, trial]

            reward_mopo = oracle_mopo(mdp.T, M, mdp.R, Vmax, gamma)  # compute the penalized reward of MOPO using the
            # oracle
            # Solve MOPO with running gamma
            inputs = [(mdp.T, reward_mopo, 0.2*i, trivial_policy) for i in range(1, 5)]
            inputs.append((mdp.T, reward_mopo, gamma, trivial_policy))
            pool = mp.Pool(mp.cpu_count())
            res_mopo = pool.starmap(policy_iteration, inputs)
            pool.close()
            pool.join()
            inputs = [(mdp.T, mdp.R, res_mopo[i], gamma, mdp.init) for i in range(len(res_mopo))]
            pool = mp.Pool(mp.cpu_count())
            res_mopo = pool.starmap(policy_eval_i, inputs)
            pool.close()
            pool.join()
            for i in range(len(res_mopo)):
                mopo[traj_number, trial, i] = res_mopo[i] / V_opt

            # Solving MOREL with kappa = 100, and running alpha from 0.1 to 0.5
            alpha_morel = [0.1 for i in range(1, morel.shape[2]+1)]
            inputs = [(100, alpha_morel[i], mdp.T, M, mdp.R, gamma, mdp.init, V_opt, trivial_policy,
                       str(config.environment)) for i in range(len(alpha_morel))]
            pool = mp.Pool(mp.cpu_count())
            res_morel = pool.starmap(solve_morel, inputs)
            pool.close()
            pool.join()

            for i in range(len(res_morel)):
                morel[traj_number, trial, i] = res_morel[i]

            # SOLVING SPIBB with different N thresholds
            Mspibb = np.zeros((mdp.T.shape[1], mdp.T.shape[0], mdp.T.shape[2]))
            Nspibb = np.zeros((mdp.T.shape[1], mdp.T.shape[0], mdp.T.shape[2]))
            Rspibb = np.mean(mdp.R, axis=2).T

            for s in range(mdp.T.shape[1]):
                for a in range(mdp.T.shape[0]):
                    for s2 in range(mdp.T.shape[2]):
                        Mspibb[s, a, s2] = M[a, s, s2]
                        Nspibb[s, a, s2] = N[a, s, s2]
            #print("SPIBB")

            mdpspibb = MDP(S=mdp.states, A=mdp.actions, T=Mspibb, R=Rspibb, gamma=gamma, temperature=0)

            spibbcount = 0
            for N_threshold in [1, 2, 3, 5, 7, 10, 20]:
                pi_spibb = SPIBB(mdpspibb, pi_b, Nspibb, N_threshold=N_threshold)
                V_spibb = policy_eval_i_s(mdp.T, mdp.R, pi_spibb, gamma, mdp.init) / V_opt
                #print(N_threshold, V_spibb)
                spibb0[traj_number, trial, spibbcount] = V_spibb
                spibbcount += 1

            # BOPAH
            fcount = 0
            for fold in [2]:  # [2, 5]:
                dcount = 0
                for dof in [20]:  # [1, 2, 4, 20, 50]:
                    alpha = Alpha(S=mdp.states, D=dof,
                                  psi=np.clip(np.ones(dof) * 1.0 / len(trajectories), 0.0001, np.inf))
                    pi_bopah, _ = BOPAH(mdp.states, mdp.actions, Rspibb, gamma, 0, trajectories, pi_b,
                                        alpha, N_folds=fold)
                    perf_bopah = policy_eval_i_s(mdp.T, mdp.R, pi_bopah, gamma, mdp.init) / V_opt
                    #print('Bopah')
                    #print(fold, dof, perf_bopah)
                    bopah0[traj_number, trial, fcount, dcount] = perf_bopah
                    dcount += 1
                fcount += 1

            np.savez_compressed(str(config.environment)+'_temp_'+str(config.number)+'.npz', t=trivial, l=lmdp, q=qlmdp,
                                m=mopo, c=clmdp, o=olmdp, r=morel, s=spibb0, b=bopah0, allow_pickle=True)
    np.savez_compressed(str(config.environment)+'_final_'+str(config.number)+'.npz', t=trivial, l=lmdp, q=qlmdp, m=mopo,
                        c=clmdp, o=olmdp, r=morel, s=spibb0, b=bopah0,  allow_pickle=True)
