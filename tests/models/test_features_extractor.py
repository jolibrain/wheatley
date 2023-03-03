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


def test_forward(gym_observation, features_extractor):
    gym_observation["features"] = gym_observation["features"].to(torch.device("cpu")).float()
    gym_observation["edge_index"] = gym_observation["edge_index"].to(torch.device("cpu")).long()
    gym_observation["mask"] = gym_observation["mask"].to(torch.device("cpu")).float()
    features = features_extractor(gym_observation)
    assert list(features.shape) == [2, 9, (8 + 4 * 64) * 2]
