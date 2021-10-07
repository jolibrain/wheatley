import numpy as np
import torch

from env.tassel_reward_model import TasselRewardModel
from utils.env_observation import EnvObservation


def test_evaluate():
    rm = TasselRewardModel(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[5, 5, 5], [6, 6, 6], [7, 7, 7]]))
    obs = EnvObservation(
        3,
        3,
        9,
        2,
        torch.tensor([[1, 5], [1, 10], [0, 15], [1, 11], [1, 17], [0, 23], [1, 18], [0, 25], [0, 32]]),
        torch.tensor([[1, 2], [2, 1]]),
        torch.zeros(81),
    )
    next_obs = EnvObservation(
        3,
        3,
        9,
        2,
        torch.tensor([[1, 5], [1, 10], [0, 15], [1, 11], [1, 17], [1, 30], [1, 18], [0, 25], [0, 34]]),
        torch.tensor([[1, 2], [2, 1]]),
        torch.zeros(81),
    )
    assert rm.evaluate(obs, None, next_obs) == -3
