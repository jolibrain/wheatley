import torch
from torch_geometric.data import Data, DataLoader

from models.mlp_extractor import MLPExtractor

from config import DEVICE, HIDDEN_DIM_FEATURES_EXTRACTOR, MAX_N_NODES


def test_compute_possible_s_a_pairs():
    me = MLPExtractor()
    graph_embedding = torch.tensor([[[0, 1]], [[2, 3]]])
    nodes_embedding = torch.tensor([[[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    assert torch.eq(
        me._compute_possible_s_a_pairs(graph_embedding, nodes_embedding),
        torch.tensor(
            [
                [
                    [0, 1, 4, 5, 4, 5],
                    [0, 1, 4, 5, 6, 7],
                    [0, 1, 6, 7, 4, 5],
                    [0, 1, 6, 7, 6, 7],
                ],
                [
                    [2, 3, 8, 9, 8, 9],
                    [2, 3, 8, 9, 10, 11],
                    [2, 3, 10, 11, 8, 9],
                    [2, 3, 10, 11, 10, 11],
                ],
            ]
        ),
    ).all()


def test_apply_mask():
    me = MLPExtractor()
    tensor = torch.rand(
        (4, 9, HIDDEN_DIM_FEATURES_EXTRACTOR),
        dtype=torch.float32,
        device=DEVICE,
    )
    mask = torch.tensor(
        [
            [0, 1, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
        ]
    )
    masked_tensor, indexes = me._apply_mask(tensor, mask)
    assert (masked_tensor[0] == tensor[0][[1, 2, 7]]).all()
    assert (masked_tensor[1] == tensor[1][[5, 6, 7]]).all()
    assert (
        masked_tensor[2]
        == torch.cat(
            [tensor[2][[7]], torch.zeros((2, HIDDEN_DIM_FEATURES_EXTRACTOR))],
            dim=0,
        )
    ).all()
    assert (
        masked_tensor[3]
        == torch.cat(
            [
                tensor[3][[6, 7]],
                torch.zeros((1, HIDDEN_DIM_FEATURES_EXTRACTOR)),
            ],
            dim=0,
        )
    ).all()


def test_forward():
    me = MLPExtractor()
    features = torch.cat(
        [
            torch.rand(
                (1, 5, HIDDEN_DIM_FEATURES_EXTRACTOR),
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                [
                    [
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
        ],
        axis=2,
    )
    pi, value = me(features)
    assert list(value.shape) == [1, 1, 1]
    assert list(pi.shape) == [1, MAX_N_NODES ** 2]
    # Check that the 0 corresponding to null nodes are in the right positions
    assert (pi[:, 4:MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + MAX_N_NODES : 2 * MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + 2 * MAX_N_NODES : 3 * MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + 3 * MAX_N_NODES : 4 * MAX_N_NODES] == 0).all()

    features = torch.tensor(
        [
            [
                [1 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [0, 0, 0, 0],
                [2 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [1, 0, 0, 0],
                [3 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [0, 0, 0, 0],
                [4 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [0, 0, 1, 0],
                [5 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [0, 0, 0, 0],
            ],
            [
                [1 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [0, 0, 0, 0],
                [2 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [0, 0, 0, 0],
                [3 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [0, 0, 0, 0],
                [4 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [0, 0, 0, 0],
                [5 for i in range(HIDDEN_DIM_FEATURES_EXTRACTOR)]
                + [0, 0, 0, 1],
            ],
        ],
        dtype=torch.float,
        device=DEVICE,
    )
    pi, value = me(features)
    assert pi[0, 0] != 0
    assert pi[0, 2 * MAX_N_NODES + 2] != 0
    pi[0, 0] = 0
    pi[0, 2 * MAX_N_NODES + 2] = 0
    assert (pi[0] == 0).all()
    assert pi[1, 4 * MAX_N_NODES + 4] != 0
    pi[1, 4 * MAX_N_NODES + 4] = 0
    assert (pi == 0).all()
