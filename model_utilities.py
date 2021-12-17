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

import numpy as np
import multiprocessing as mp
from math import floor
from scipy.stats import binom
from solvers import policy_eval_i


def single_sample(alphas):
    """
    Generate a samples from an array of alpha distributions.
    """
    r = np.random.standard_gamma(alphas)
    return r / r.sum(-1, keepdims=True)


def dirichlet_sample(alphas, number):
    """
    Generate samples from an array of alpha distributions.
    """
    r = np.zeros((number, alphas.shape[0], alphas.shape[1], alphas.shape[2]))
    for n in range(number):
        r[n] = single_sample(alphas)
    return r


# FUNCTION THAT SUMS THE BINOMIAL PMF WITH PARAMETERS n and p BETWEEN INDICES r and s
def binom_sum(r, s, n, p):
    o = 0
    for i in range(r, s):
        o += binom.pmf(i, n, p)
    return o


# FUNCTION THAT ITERATIVELY SAMPLES MODELS FROM THE POSTERIOR, PERFORMS POLICY EVALUATION AND ESTIMATES
# THE REQUIRED QUANTILE OF THE DISTRIBUTION OF THE PERFORMANCE OF THE POLICY
# FINALLY RETURNS AS AN OUTPUT THE EXP. VALUE, THE VaR-quantile, CVaR-quantile and Optimistic-quantile OVER THE
# EMPIRICAL DISTRIBUTION.
# Input: policy to evaluate, dirichlet sampling function, parameters of the dirichlet distribution, most likely model,
#         immediate reward, gamma, initial distribution, quantile to evaluate, size of the batch of models to sample
#         at each iteration, relative tolerance, confidence threshold, max number of models to sample
# Output: (Expected value, policy, VaR-quantile, CVaR-quantile, Optimistic-quantile)
def quantile_calc(policy, sampler, alpha, M, R, gamma, init, quantile, batch_size, rel_tol, evaluator=policy_eval_i,
                  confidence=0.01, max_size=4000):
    M = M.reshape((1, M.shape[0], M.shape[1], M.shape[1]))
    sample = sampler(alpha, batch_size)
    sample = np.vstack((sample, M))
    f_sample = []
    pool = mp.Pool(mp.cpu_count())
    result = pool.starmap_async(evaluator, [(model, R, policy, gamma, init) for model in sample])
    pool.close()
    pool.join()
    res = result.get()
    for value in res:
        f_sample.append(value)
    del result, res
    done = False
    f_sample.sort()
    while not done and len(f_sample) < max_size:
        n = len(f_sample)
        print(policy, n, end='         \r')
        r, s = binom.interval(0.5, n, quantile)
        r = int(r)
        s = int(s)
        c = 0
        prob = binom_sum(r, s, n, quantile)
        first_zero = False
        first_endlist = False
        while (r != 0 or s != n-1) and not done:
            n = len(f_sample)
            if prob > 1-confidence and f_sample[s]-f_sample[r] < rel_tol*(f_sample[-1]-f_sample[0]):
                done = True
            else:
                c = (c+1) % 2
                if c == 0:
                    r = max(0, r-1)
                    if first_zero is False:
                        prob += binom.pmf(r, n, quantile)
                    if r == 0:
                        first_zero = True
                if c == 1:
                    s = min(n-1, s+1)
                    if first_endlist is False:
                        prob += binom.pmf(s, n, quantile)
                    if s == n-1:
                        first_endlist = True
        sample = sampler(alpha, batch_size)
        pool = mp.Pool(mp.cpu_count())
        result = pool.starmap_async(evaluator, [(model, R, policy, gamma, init) for model in sample])
        pool.close()
        pool.join()
        res = result.get()
        for value in res:
            f_sample.append(value)
        del result, res
        f_sample.sort()
    output = [np.mean(f_sample), policy, f_sample[floor(len(f_sample)*quantile)],
              np.mean(f_sample[:floor(len(f_sample)*quantile)]), np.mean(f_sample[floor(len(f_sample)*quantile):])]
    return output


# FUNCTION THAT ESTIMATES THE MOST LIKELY TRANSITION MATRIX OF A MDP
# Input: true MDP transitions (just to extract the dimensions), batch of trajectories
# Output: Most likely transition matrix M, transition counter N
def learn_transition(P, his):
    M = np.zeros(P.shape, dtype=np.float64)
    sommaM = np.zeros((P.shape[0], P.shape[1]), dtype=np.float64)
    for i in range(len(his)):
        M[his[i][1], his[i][0], his[i][-1]] += 1
    N = np.copy(M)
    for j in range(P.shape[1]):
        for a in range(P.shape[0]):
            sommaM[a, j] = np.sum(M[a, j], dtype=np.float64)
            if sommaM[a, j] != 0:
                M[a, j] /= sommaM[a, j]
            else:
                M[a, j] = 1./P.shape[1]*np.ones(P.shape[1])
    return M, N


# FUNCTION THAT ESTIMATES THE MOST LIKELY TRANSITION MATRIX OF A MULTI ARMED BANDIT
# Input: true MAB transitions (just to extract the dimensions), batch of trajectories
# Output: Most likely transition matrix M, transition counter N
def learn_mab(P, his):
    M = np.zeros(P.shape, dtype= np.float64)
    sommaM = np.zeros(P.shape[0], dtype=np.float64)
    for i in range(len(his)):
        M[his[i][0], his[i][1]] += 1
    N = np.copy(M)
    for a in range(P.shape[0]):
        sommaM[a] = np.sum(M[a], dtype=np.float64)
        if sommaM[a] != 0:
            M[a] /= sommaM[a]
        else:
            M[a] = 1./P.shape[1]*np.ones(P.shape[1])
    return M, N


# FUNCTION THAT GENERATES THE TRANSITION AND REWARD FUNCTION OF THE PENALIZED MDP
# USING THE UNCERTAINTY GIVEN BY AN ORACLE.
# Input: true transition T, most likely transition M, immediate reward R, Maximum Value of the Optimal Value function
# Output: Penalized MDP immediate reward
def oracle_mopo(T, M, R, Vmax, gamma):
    diff = np.abs(T-M)
    reward = np.zeros_like(R)
    error = np.zeros((M.shape[0], M.shape[1]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            error[i, j] = 0.5 * np.sum(diff[i, j])  # Total Variation Distance
            for k in range(M.shape[2]):
                reward[i, j, k] = R[i, j, k] - Vmax * gamma * error[i, j]
    return reward


# FUNCTION THAT GENERATES THE TRANSITION AND REWARD FUNCTION OF THE PESSIMISTIC (kappa, alpha) MDP
# USING THE UNCERTAINTY GIVEN BY AN ORACLE.
# Input: Penalization kappa, uncertainty threshold alpha, true transition T, most likely transition M,
#        immediate reward R
# Output: Pessimistic MDP transition M and pessimistic MDP immediate reward R
def oracle_morel(kappa, alpha, T, M, R):
    error = np.zeros((M.shape[0], M.shape[1]))
    diff = np.abs(T - M)
    reward = np.zeros((R.shape[0], R.shape[1]+1, R.shape[2]+1))
    M_morel = np.zeros((M.shape[0], M.shape[1]+1, M.shape[1]+1))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            error[i, j] = 0.5*np.sum(diff[i, j])  # Total Variation Distance
            if error[i, j] > alpha:
                M_morel[i, j] = np.eye(M.shape[1]+1)[-1]
                for k in range(M.shape[2]):
                    reward[i, j, k] = -kappa
            else:
                M_morel[i, j, :-1] = M[i, j]
                for k in range(M.shape[2]):
                    reward[i, j, k] = R[i, j, k]
        M_morel[i, -1, -1] = 1
        reward[i, -1, :] = -kappa
    return M_morel, reward


# FUNCTION THAT CHECKS IF A MATRIX IS INVERTIBLE
def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


