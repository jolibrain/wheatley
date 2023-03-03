#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np

from env.env import Env
from sb3_contrib.common.maskable.utils import get_action_masks


class RandomAgent:
    def __init__(self, max_n_jobs, max_n_machines):
        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.max_n_nodes = self.max_n_jobs * self.max_n_machines

    def predict(self, env):
        # soft reset to evaluate the same sampled problem as PPO
        observation = env.reset(soft=True)
        done = False
        while not done:
            action = self.select_action(env)
            observation, _, done, _ = env.step(action)
        solution = env.get_solution()
        return solution

    def select_action(self, env):
        action_masks = get_action_masks(env)
        possible_actions = np.nonzero(action_masks)[0]
        action = np.random.choice(possible_actions)
        return action
