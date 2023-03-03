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

import torch
import torch.nn as nn

from env.reward_model import RewardModel


class IntrinsicRewardModel(RewardModel):
    def __init__(self, observation_input_size, n_nodes):
        self.random_network = nn.Sequential(
            nn.Linear(observation_input_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16), nn.Sigmoid()
        )
        self.predictor_network = nn.Sequential(
            nn.Linear(observation_input_size, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.Sigmoid()
        )
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=0.01)
        self.n_nodes = n_nodes

    def evaluate(self, obs, action, next_obs):
        inp = obs.features.flatten()
        self.optimizer.zero_grad()
        output = self.predictor_network(inp)
        target = self.random_network(inp)
        target = torch.clip(target, -0.5, 0.5)
        loss = self.criterion(output, target)
        reward = loss.item()
        loss.backward()
        self.optimizer.step()
        reward = reward / self.n_nodes  # We divide by the number of nodes to have a return which is between -1 and 1
        return reward
