from gym.spaces import Dict, Box, Discrete
import numpy as np
import torch

from models.features_extractor import FeaturesExtractor

from config import DEVICE


def test_forward(gym_observation):
    fe = FeaturesExtractor(
        observation_space=Dict(
            {
                "n_nodes": Discrete(9),
                "features": Box(0, 1, shape=(9, 2)),
                "edge_index": Box(0, 9, shape=(2, 81), dtype=np.int64),
                "mask": Box(0, 1, shape=(81,), dtype=np.int64),
            }
        )
    )
    gym_observation["features"] = (
        gym_observation["features"].to(DEVICE).float()
    )
    gym_observation["edge_index"] = (
        gym_observation["edge_index"].to(DEVICE).long()
    )
    gym_observation["mask"] = gym_observation["mask"].to(DEVICE).float()
    features = fe(gym_observation)
    assert list(features.shape) == [1, 10, 41]
    assert list(features[0, 1, 32:35].detach().cpu().numpy()) == [
        1,
        1,
        1,
    ]  # TODO comprendre
