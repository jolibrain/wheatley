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

import pytest

import numpy as np
from stable_baselines3.common.env_checker import check_env


def test_observation_shape_and_validity(env):
    obs = env.reset()
    assert obs["n_nodes"] == 25
    assert list(obs["features"].shape) == [
        25,
        12,
    ]
    assert list(obs["edge_index"].shape) == [2, 625]
    assert not np.isnan(obs["features"]).any()
    assert not np.isnan(obs["edge_index"]).any()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_check_env(env):
    check_env(env)
