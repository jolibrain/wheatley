from gym.spaces import Dict, Box, Discrete
import numpy as np
import torch

from models.features_extractor import FeaturesExtractor


def test_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = FeaturesExtractor(
        observation_space=Dict(
            {
                "n_nodes": Discrete(9),
                "features": Box(0, 1, shape=(9, 2)),
                "edge_index": Box(0, 9, shape=(2, 81), dtype=np.int64),
            }
        )
    )
    observation = {
        "n_nodes": torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], device=device),
        "features": torch.tensor(
            [
                [
                    [0.6, 0.3],
                    [0.4, 0.4],
                    [0.8, 0.1],
                    [0.2, 0.1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        ),
        "edge_index": torch.tensor(
            [[[1, 3, 2, 0, 1, 2], [2, 1, 3, 3, 2, 0]]], dtype=torch.int64, device=device
        ),
    }
    features = fe(observation)
    assert list(features.shape) == [1, 5, 32]
