from gym.spaces import Dict, Box, Discrete
import numpy as np
import torch

from models.features_extractor import FeaturesExtractor


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
    features = fe(gym_observation)
    assert list(features.shape) == [1, 10, 41]
    assert list(features[0, 1, 32:35].detach().numpy()) == [
        1,
        1,
        1,
    ]  # TODO comprendre
