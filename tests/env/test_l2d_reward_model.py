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

from copy import deepcopy


from env.reward_models.l2d_reward_model import L2DRewardModel


def test_evaluate(env_observation):
    rm = L2DRewardModel()
    next_env_observation = deepcopy(env_observation)
    next_env_observation.features[0, 4:8] = 10
    next_env_observation.features[4, 4:8] = 14
    assert rm.evaluate(env_observation, None, next_env_observation) == 1
    next_env_observation.features[4, 4:8] = 10
    next_env_observation.features[5, 4:8] = 10
    assert rm.evaluate(env_observation, None, next_env_observation) == 5
