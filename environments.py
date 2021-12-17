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
import gym


# ENVIRONMENT CLASS, CONTAINS THE FOLLOWING VARIABLES
# self.actions = size of the action space A
# self.states = size of the state space S
# self.T = transition matrix AxSxS -> [0,1]
# self.R = immediate reward matrix AxSxS -> R+
# self.init = initial state distribution
# Also contains the method: generate_trajectory(number, length) which generate number trajectories
# with a fixed length while following a random policy.
class Ring:
    def __init__(self):
        self.actions = 3
        self.states = 5
        self.T = np.zeros((3, 5, 5))
        self.R = 0.1*np.ones((3, 5, 5))

        # a = 0 --- GO LEFT
        self.T[0, 0, 4] = 1
        self.T[0, 1, 0] = 1
        self.T[0, 2, 1] = 0.5
        self.T[0, 2, 2] = 0.5
        self.T[0, 3, 2] = 1
        self.T[0, 4, 3] = 0.5
        self.T[0, 4, 4] = 0.5

        # a = 1 --- STAY
        self.T[1, 0, 0] = 0.8
        self.T[1, 0, 1] = 0.1
        self.T[1, 0, 4] = 0.1
        self.T[1, 1, 0] = 0.1
        self.T[1, 1, 1] = 0.8
        self.T[1, 1, 2] = 0.1
        self.T[1, 2, 2] = 1
        self.T[1, 3, 2] = 0.1
        self.T[1, 3, 3] = 0.8
        self.T[1, 3, 4] = 0.1
        self.T[1, 4, 4] = 1

        # a = 2 --- GO RIGHT
        self.T[2, 0, 0] = 0.1
        self.T[2, 0, 1] = 0.9
        self.T[2, 1, 1] = 0.1
        self.T[2, 1, 2] = 0.9
        self.T[2, 2, 2] = 0.5
        self.T[2, 2, 3] = 0.5
        self.T[2, 3, 3] = 0.1
        self.T[2, 3, 4] = 0.9
        self.T[2, 4, 0] = 0.5
        self.T[2, 4, 4] = 0.5

        # REWARDS
        self.R[0, 2, 1] = 0
        self.R[0, 2, 2] = 0
        self.R[1, 2, 2] = 0
        self.R[2, 2, 2] = 0
        self.R[2, 2, 3] = 0.5
        self.R[0, 4, 3] = 0.5
        self.R[0, 4, 4] = 0
        self.R[1, 3, 3] = 1
        self.R[2, 3, 3] = 1
        self.R[2, 4, 0] = 0
        self.R[2, 4, 4] = 0

        self.init = np.eye(self.states, dtype=np.float64)[0]

    #
    def generate_trajectories(self, number, length):
        history = []
        trajectories = []
        init = np.zeros(self.states, dtype=np.float64)
        for i in range(number):
            temp_traj = []
            state = np.random.choice(self.states, p=self.init)
            init[state] += 1
            for j in range(length):
                a = np.random.randint(0, self.actions)
                new_state = np.random.choice(range(self.states), p=self.T[a, state])
                r = self.R[a, state, new_state]
                history.append([state, a, r, new_state])
                temp_traj.append([i, j, state, a, r, new_state])
                state = new_state
            trajectories.append(temp_traj)
        init /= np.sum(init)
        batch_traj = [val for sublist in trajectories for val in sublist]
        return history, init, trajectories, batch_traj


# inherits method from Ring
class Chain(Ring):
    def __init__(self):
        self.states = 5
        self.actions = 2
        self.R = np.zeros((self.actions, self.states, self.states))
        self.T = np.zeros_like(self.R)
        self.init = np.eye(self.states)[0]
        self.T[0, :, 0] = 0.2
        self.T[1, :, 0] = 0.8
        self.T[0, -1, -1] = 0.8
        self.T[1, -1, -1] = 0.2
        self.R[1, :, :] = 2.
        self.R[0, -1, -1] = 10.
        for s in range(self.states-1):
            self.T[0, s, s+1] = 0.8
            self.T[1, s, s+1] = 0.2


# inherits method from Ring
class FrozenLake(Ring):
    def __init__(self, type='random'):
        self.env = gym.make('FrozenLake8x8-v1')
        self.states = 64
        self.actions = 4
        description = self.env.P
        self.init = self.env.isd
        self.T = np.zeros((self.actions, self.states, self.states))
        self.R = np.zeros_like(self.T)
        ending = []
        for a in range(self.actions):
            for s in range(self.states):
                for i in range(len(description[s][a])):
                    s_new = description[s][a][i][1]
                    reward = description[s][a][i][2]
                    done = description[s][a][i][-1]
                    self.R[a, s, s_new] = reward
                    if str(type) == 'random':
                        if reward == 0 and not done:
                            self.R[a, s, s_new] = np.random.uniform(0, 0.8)
                    if s not in ending:
                        self.T[a, s, s_new] = np.random.uniform(0, 100)
                    if done:
                        ending.append(s_new)
                        self.T[a, s_new] = np.eye(self.states)[s_new]
        for a in range(self.actions):
            for s in range(self.states):
                somma = np.sum(self.T[a, s])
                if somma > 0:
                    self.T[a, s] /= somma
