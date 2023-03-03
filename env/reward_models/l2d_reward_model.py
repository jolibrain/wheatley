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

from env.reward_model import RewardModel


class L2DRewardModel(RewardModel):
    def __init__(self):
        pass

    def evaluate(self, obs, action, next_obs):
        """
        Reward is computed as H(s_t) - H(s_t+1) where H(s_t) = max{tasks_lower_bound} at
        time s_t. For more info, see https://arxiv.org/abs/2010.12367
        """
        features_t = obs.features
        features_tp = next_obs.features

        H_st = torch.max(features_t[:, 1]).item()
        if H_st < 0:
            H_st = 0
        H_stp = torch.max(features_tp[:, 1]).item()
        if H_stp < 0:
            H_stp = 0
        reward = H_st - H_stp

        return reward
