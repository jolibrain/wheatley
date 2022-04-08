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
        torch.zeros(9),
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
        torch.zeros(9),
        3,
        3,
    )
    assert rm.evaluate(obs, None, next_obs) == -3
