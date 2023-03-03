#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import torch


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
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                        ]
                    ],
                    device=torch.device("cpu"),
                ).float(),
                "edge_index": torch.tensor(
                    [
                        [
                            [0, 2, 3, 4, 6, 7] + [0 for i in range(75)],
                            [1, 1, 4, 5, 7, 8] + [0 for i in range(75)],
                        ]
                    ],
                    device=torch.device("cpu"),
                ).long(),
                "mask": torch.zeros((1, 9), device=torch.device("cpu")),
            },
            torch.tensor([0], device=torch.device("cpu")).float(),
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
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                        ]
                    ],
                    device=torch.device("cpu"),
                ).float(),
                "edge_index": torch.tensor(
                    [
                        [
                            [0, 1, 3, 4, 6, 7] + [0 for i in range(75)],
                            [2, 2, 4, 5, 7, 8] + [0 for i in range(75)],
                        ]
                    ],
                    device=torch.device("cpu"),
                ).long(),
                "mask": torch.zeros((1, 9), device=torch.device("cpu")),
            },
            torch.tensor([1], device=torch.device("cpu")).float(),
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
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                        ]
                    ],
                    device=torch.device("cpu"),
                ).float(),
                "edge_index": torch.tensor(
                    [
                        [
                            [0, 1, 3, 4, 6, 7] + [0 for i in range(75)],
                            [1, 2, 4, 5, 7, 8] + [0 for i in range(75)],
                        ]
                    ],
                    device=torch.device("cpu"),
                ).long(),
                "mask": torch.zeros((1, 9), device=torch.device("cpu")),
            },
            torch.tensor([0], device=torch.device("cpu")).float(),
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
                            [1, 1, 1, 1, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                            [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
                        ]
                    ],
                    device=torch.device("cpu"),
                ).float(),
                "edge_index": torch.tensor(
                    [
                        [
                            [0, 1, 3, 4, 6, 7] + [0 for i in range(75)],
                            [1, 2, 4, 5, 7, 8] + [0 for i in range(75)],
                        ]
                    ],
                    device=torch.device("cpu"),
                ).long(),
                "mask": torch.zeros((1, 9), device=torch.device("cpu")),
            },
            torch.tensor([1], device=torch.device("cpu")).float(),
        ),
    ]
    last_layer = torch.nn.Linear(9 * 64, 1)
    last_layer.to(torch.device("cpu"))
    optimizer = torch.optim.Adam(features_extractor.parameters())
    criterion = torch.nn.BCELoss()

    for epoch in range(100):
        for x, y in dataset:
            embedded_features = features_extractor(x).squeeze()
            prediction = torch.sigmoid(last_layer(embedded_features[:, 0:64].flatten().squeeze()))
            criterion.zero_grad()
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

    for x, y in dataset:
        embedded_features = features_extractor(x).squeeze()
        prediction = torch.sigmoid(last_layer(embedded_features[:, 0:64].flatten().squeeze()))
        if torch.cuda.is_available():
            prediction = prediction.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        else:
            prediction = prediction.detach().numpy()
            y = y.detach().numpy()
        assert np.round(prediction) == y
