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
