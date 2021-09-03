import pytest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from env.env import Env
from problem.problem_description import ProblemDescription

from config import MAX_N_NODES, MAX_N_EDGES


def test_observation_shape_and_validity(env):
    obs = env.reset()
    assert obs["n_nodes"] == 25
    assert list(obs["features"].shape) == [MAX_N_NODES, 2]
    assert list(obs["edge_index"].shape) == [2, MAX_N_EDGES]
    assert not np.isnan(obs["features"]).any()
    assert not np.isnan(obs["edge_index"]).any()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_check_env(env):
    check_env(env)
