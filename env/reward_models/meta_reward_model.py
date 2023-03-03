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

from env.reward_model import RewardModel


class MetaRewardModel(RewardModel):
    def __init__(self, reward_model_cls_list, reward_model_kwargs_list, coefs, n_timesteps):
        self.reward_models = [
            reward_model_cls_list[i](**reward_model_kwargs_list[i]) for i in range(len(reward_model_cls_list))
        ]
        self.coefs = coefs
        self.cur_coefs = coefs
        self.n_timesteps = n_timesteps

    def evaluate(self, obs, action, next_obs):
        rewards = [self.reward_models[i].evaluate(obs, action, next_obs) for i in range(len(self.reward_models))]
        reward = 0
        for i in range(len(self.reward_models)):
            reward += self.cur_coefs[i] * rewards[i]
        self.cur_coefs[0] = min(1, self.cur_coefs[0] + (1 - self.coefs[0]) / self.n_timesteps)
        self.cur_coefs[1] = max(0, self.cur_coefs[1] - self.coefs[1] / self.n_timesteps)
        return reward
