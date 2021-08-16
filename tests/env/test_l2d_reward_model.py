import torch

from env.l2d_reward_model import L2DRewardModel
from utils.state import State


def test_evaluate():
    rm = L2DRewardModel()
    features_1 = torch.tensor([[[0, 3], [0, 1], [1, 14], [1, 1]]])
    features_2 = torch.tensor([[[1, 2], [0, 6], [1, 7], [1, 3]]])
    assert (
        rm.evaluate(State(4, features_1, None), 0, State(4, features_2, None))
        == 7
    )
