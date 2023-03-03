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


class SparseRewardModel(RewardModel):
    def __init__(self):
        pass

    def evaluate(self, obs, action, next_obs):
        """
        Reward is 0 for every time steps, except for the last one, where it is the opposite of the Makespan
        """
        features_tp = next_obs.features
        # below everything seleced, ok only for jssp, not for -1 in task affectration
        # is_done = (features_tp[:, 0] == 1).all().item()
        # nothing selectable anymore
        is_done = (features_tp[:, 5] == 0).all().item()
        if not is_done:
            return 0
        makespan = torch.max(features_tp[:, 1]).item()
        # We don't want |reward| to be > 1. Since makespan is divided by longest task, we only have to divide by 2
        reward = makespan / 2
        return -reward
