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
import torch


class RandomAgent:
    def __init__(self, nonchrono):
        self.nonchrono = nonchrono

    def predict(self, env):
        # soft reset to evaluate the same sampled problem as PPO
        observation, info = env.reset(soft=True)
        action_mask = info["mask"]
        done = False
        while not done:
            action = self.select_action(env, action_mask)
            observation, _, done, _, info = env.step(action)
            action_mask = info["mask"]
        solution = env.get_solution()
        return solution

    def select_action(self, env, action_masks):
        if self.nonchrono == "path":
            return (torch.rand(env.problem.ncells) * 2 - 1.0).unsqueeze(-1)
        if self.nonchrono == "order":
            return (
                (torch.rand(env.env_specification.max_order) * env.problem.ncells)
                .int()
                .unsqueeze(-1)
            )

        if isinstance(action_masks, torch.Tensor):
            mask = action_masks.detach().cpu().numpy()
        else:
            mask = np.asarray(action_masks)

        if mask.ndim == 1:
            possible_actions = np.nonzero(mask)[0]
            return np.random.choice(possible_actions)

        actions = []
        for agent_mask in mask:
            possible_actions = np.nonzero(agent_mask)[0]
            if len(possible_actions) == 0:
                actions.append(0)
            else:
                actions.append(int(np.random.choice(possible_actions)))
        return np.array(actions)
