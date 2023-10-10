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

from generic.reward_model import RewardModel
from jssp.utils.utils import node_to_job_and_task


class TasselRewardModel(RewardModel):
    def __init__(self, affectations, durations, normalize_input):
        self.affectations = affectations
        self.durations = durations
        self.dividing_factor = (
            np.sum(self.durations.flatten()) if normalize_input else 1
        )

    def evaluate(self, obs, action, next_obs):
        """
        See https://arxiv.org/pdf/2104.03760.pdf for reward implementation explaination
        """
        features_t = obs.features
        features_tp = next_obs.features
        scheduled_node_id = (
            (features_tp[:, 0] - features_t[:, 0]).nonzero(as_tuple=True)[0].item()
        )
        job_id, task_id = node_to_job_and_task(
            scheduled_node_id, self.affectations.shape[1]
        )
        cur_duration = self.durations[job_id, task_id]

        ancient_idle_time = torch.sum(
            torch.max(features_t[:, 7].reshape(self.affectations.shape), axis=1).values
        )
        new_idle_time = torch.sum(
            torch.max(features_tp[:, 7].reshape(self.affectations.shape), axis=1).values
        )
        reward = (cur_duration / self.dividing_factor) - (
            new_idle_time - ancient_idle_time
        )
        return reward.item()
