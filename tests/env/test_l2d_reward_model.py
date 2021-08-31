from copy import deepcopy

import torch

from env.l2d_reward_model import L2DRewardModel


def test_evaluate(observation):
    rm = L2DRewardModel()
    next_observation = deepcopy(observation)
    next_observation.features[0, 0, 2] = 10
    next_observation.features[0, 4, 2] = 14
    assert rm.evaluate(observation, None, next_observation) == 1
    next_observation.features[0, 4, 2] = 10
    next_observation.features[0, 5, 2] = 10
    assert rm.evaluate(observation, None, next_observation) == 5
