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

from env.reward_models.tassel_reward_model import TasselRewardModel
from utils.env_observation import EnvObservation


def test_evaluate():
    rm = TasselRewardModel(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[5, 5, 5], [6, 6, 6], [7, 7, 7]]), False)
    obs = EnvObservation(
        3,
        3,
        torch.tensor(
            [
                [1, 1, 1, 1, 5, 5, 5, 5],
                [1, 1, 1, 1, 10, 10, 10, 10],
                [0, 0, 0, 0, 15, 15, 15, 15],
                [1, 1, 1, 1, 11, 11, 11, 11],
                [1, 1, 1, 1, 17, 17, 17, 17],
                [0, 0, 0, 0, 23, 23, 23, 23],
                [1, 1, 1, 1, 18, 18, 18, 18],
                [0, 0, 0, 0, 25, 25, 25, 25],
                [0, 0, 0, 0, 32, 32, 32, 32],
            ]
        ),
        torch.tensor([[1, 2], [2, 1]]),
        3,
        3,
    )
    next_obs = EnvObservation(
        3,
        3,
        torch.tensor(
            [
                [1, 1, 1, 1, 5, 5, 5, 5],
                [1, 1, 1, 1, 10, 10, 10, 10],
                [0, 0, 0, 0, 15, 15, 15, 15],
                [1, 1, 1, 1, 11, 11, 11, 11],
                [1, 1, 1, 1, 17, 17, 17, 17],
                [1, 1, 1, 1, 30, 30, 30, 30],
                [1, 1, 1, 1, 18, 18, 18, 18],
                [0, 0, 0, 0, 25, 25, 25, 25],
                [0, 0, 0, 0, 34, 34, 34, 34],
            ]
        ),
        torch.tensor([[1, 2], [2, 1]]),
        3,
        3,
    )
    assert rm.evaluate(obs, None, next_obs) == -3
