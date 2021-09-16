import numpy as np
import torch

from models.features_extractor import FeaturesExtractor

from config import HIDDEN_DIM_FEATURES_EXTRACTOR, DEVICE


def test_features_extractor_learning(features_extractor):
    dataset = [
        (
            {
                "n_jobs": 3,
                "n_machines": 3,
                "n_nodes": 9,
                "n_edges": 6,
                "features": torch.tensor(
                    [
                        [
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                        ]
                    ],
                    device=DEVICE,
                ).float(),
                "edge_index": torch.tensor(
                    [
                        [
                            [0, 2, 3, 4, 6, 7] + [0 for i in range(75)],
                            [1, 1, 4, 5, 7, 8] + [0 for i in range(75)],
                        ]
                    ],
                    device=DEVICE,
                ).long(),
                "mask": torch.zeros((1, 81), device=DEVICE),
            },
            torch.tensor([0], device=DEVICE).float(),
        ),
        (
            {
                "n_jobs": 3,
                "n_machines": 3,
                "n_nodes": 9,
                "n_edges": 6,
                "features": torch.tensor(
                    [
                        [
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                        ]
                    ],
                    device=DEVICE,
                ).float(),
                "edge_index": torch.tensor(
                    [
                        [
                            [0, 1, 3, 4, 6, 7] + [0 for i in range(75)],
                            [2, 2, 4, 5, 7, 8] + [0 for i in range(75)],
                        ]
                    ],
                    device=DEVICE,
                ).long(),
                "mask": torch.zeros((1, 81), device=DEVICE),
            },
            torch.tensor([1], device=DEVICE).float(),
        ),
        (
            {
                "n_jobs": 3,
                "n_machines": 3,
                "n_nodes": 9,
                "n_edges": 6,
                "features": torch.tensor(
                    [
                        [
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                        ]
                    ],
                    device=DEVICE,
                ).float(),
                "edge_index": torch.tensor(
                    [
                        [
                            [0, 1, 3, 4, 6, 7] + [0 for i in range(75)],
                            [1, 2, 4, 5, 7, 8] + [0 for i in range(75)],
                        ]
                    ],
                    device=DEVICE,
                ).long(),
                "mask": torch.zeros((1, 81), device=DEVICE),
            },
            torch.tensor([0], device=DEVICE).float(),
        ),
        (
            {
                "n_jobs": 3,
                "n_machines": 3,
                "n_nodes": 9,
                "n_edges": 6,
                "features": torch.tensor(
                    [
                        [
                            [1, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                            [0.5, 0.3],
                        ]
                    ],
                    device=DEVICE,
                ).float(),
                "edge_index": torch.tensor(
                    [
                        [
                            [0, 1, 3, 4, 6, 7] + [0 for i in range(75)],
                            [1, 2, 4, 5, 7, 8] + [0 for i in range(75)],
                        ]
                    ],
                    device=DEVICE,
                ).long(),
                "mask": torch.zeros((1, 81), device=DEVICE),
            },
            torch.tensor([1], device=DEVICE).float(),
        ),
    ]
    last_layer = torch.nn.Linear(10 * HIDDEN_DIM_FEATURES_EXTRACTOR, 1)
    last_layer.to(DEVICE)
    optimizer = torch.optim.Adam(features_extractor.parameters())
    criterion = torch.nn.BCELoss()

    for epoch in range(100):
        for x, y in dataset:
            embedded_features = features_extractor(x).squeeze()
            prediction = torch.sigmoid(last_layer(embedded_features[:, 0:HIDDEN_DIM_FEATURES_EXTRACTOR].flatten().squeeze()))
            criterion.zero_grad()
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

    for x, y in dataset:
        embedded_features = features_extractor(x).squeeze()
        prediction = torch.sigmoid(last_layer(embedded_features[:, 0:HIDDEN_DIM_FEATURES_EXTRACTOR].flatten().squeeze()))
        if torch.cuda.is_available():
            prediction = prediction.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        else:
            prediction = prediction.detach().numpy()
            y = y.detach().numpy()
        assert np.round(prediction) == y
