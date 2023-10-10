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

from generic.reward_model import RewardModel


class UncertainRewardModel(RewardModel):
    def __init__(self, config):
        # 4 uses real completion time
        # 5 is for min
        # 6 is for max
        # 7 is for mode
        if config == "optimistic":
            self.index = 2
        elif config == "pessimistic":
            self.index = 3
        elif config == "realistic":
            self.index = 1
        elif config == "averagistic":
            self.index = 4
        else:
            raise Exception("Reward model not recognized :  " + config)

    def evaluate(self, obs, action, next_obs):
        """
        Reward is computed as H(s_t) - H(s_t+1) where H(s_t) = max{tasks_lower_bound} at
        time s_t. For more info, see https://arxiv.org/abs/2010.12367
        """
        features_t = obs.features
        features_tp = next_obs.features
        H_st = torch.max(features_t[:, self.index]).item()
        if H_st < 0:
            H_st = 0
        H_stp = torch.max(features_tp[:, self.index]).item()
        if H_stp < 0:
            H_stp = 0
        reward = H_st - H_stp
        return reward
