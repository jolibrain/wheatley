import torch

from models.mlp_extractor import MLPExtractor

from config import HIDDEN_DIM_FEATURES_EXTRACTOR, N_LAYERS_FEATURES_EXTRACTOR, MAX_N_NODES


def test_get_pairs_to_compute():
    me = MLPExtractor(False, "tanh", torch.device("cpu"), 2)
    graph_embedding = torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]])
    nodes_embedding = torch.tensor([[[10, 11], [12, 13]], [[14, 15], [16, 17]], [[18, 19], [20, 21]], [[22, 23], [24, 25]]])
    mask = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
        ]
    )
    pairs, indexes = me._get_pairs_to_compute(graph_embedding, nodes_embedding, mask)

    assert (
        pairs
        == torch.tensor(
            [
                [[0, 1, 10, 11, 12, 13], [0, 1, 12, 13, 10, 11]],
                [[2, 3, 16, 17, 16, 17], [0, 0, 0, 0, 0, 0]],
                [[4, 5, 20, 21, 18, 19], [0, 0, 0, 0, 0, 0]],
                [[6, 7, 22, 23, 24, 25], [6, 7, 24, 25, 22, 23]],
            ]
        )
    ).all()


def test_forward():
    # Without PDR Boolean
    me = MLPExtractor(False, "tanh", torch.device("cpu"), 2)
    features1 = torch.cat(
        [
            torch.rand(
                (1, 5, HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR),
                dtype=torch.float32,
                device=torch.device("cpu"),
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
                ],
                device=torch.device("cpu"),
            ),
        ],
        axis=2,
    )
    pi, value = me(features1)
    assert list(value.shape) == [1, 1, 1]
    assert list(pi.shape) == [1, MAX_N_NODES ** 2]
    # Check that the 0 corresponding to null nodes are in the right positions
    assert (pi[:, 4:MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + MAX_N_NODES : 2 * MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + 2 * MAX_N_NODES : 3 * MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + 3 * MAX_N_NODES : 4 * MAX_N_NODES] == 0).all()

    features2 = torch.tensor(
        [
            [
                [1 for i in range(2 + HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR)] + [0, 0, 0, 0],
                [2 for i in range(2 + HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR)] + [1, 0, 0, 0],
                [3 for i in range(2 + HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR)] + [0, 0, 0, 0],
                [4 for i in range(2 + HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR)] + [0, 0, 1, 0],
                [5 for i in range(2 + HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR)] + [0, 0, 0, 0],
            ],
            [
                [1 for i in range(2 + HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR)] + [0, 0, 0, 0],
                [2 for i in range(2 + HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR)] + [0, 0, 0, 0],
                [3 for i in range(2 + HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR)] + [0, 0, 0, 0],
                [4 for i in range(2 + HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR)] + [0, 0, 0, 0],
                [5 for i in range(2 + HIDDEN_DIM_FEATURES_EXTRACTOR * N_LAYERS_FEATURES_EXTRACTOR)] + [0, 0, 0, 1],
            ],
        ],
        dtype=torch.float,
        device=torch.device("cpu"),
    )
    pi, value = me(features2)
    assert pi[0, 0] != 0
    assert pi[0, 2 * MAX_N_NODES + 2] != 0
    pi[0, 0] = 0
    pi[0, 2 * MAX_N_NODES + 2] = 0
    assert (pi[0] == 0).all()
    assert pi[1, 3 * MAX_N_NODES + 3] != 0
    pi[1, 3 * MAX_N_NODES + 3] = 0
    assert (pi == 0).all()

    # Without PDR Boolean
    me = MLPExtractor(True, "tanh", torch.device("cpu"), 2)
    pi, value = me(features1)
    assert list(pi.shape) == [1, 2 * MAX_N_NODES ** 2]
    # Check that the 0 corresponding to null nodes are in the right positions
    assert (pi[:, 4:MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + MAX_N_NODES : 2 * MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + 2 * MAX_N_NODES : 3 * MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + 3 * MAX_N_NODES : 4 * MAX_N_NODES] == 0).all()

    assert (pi[:, MAX_N_NODES ** 2 + 4 : MAX_N_NODES ** 2 + MAX_N_NODES] == 0).all()
    assert (pi[:, MAX_N_NODES ** 2 + MAX_N_NODES + 4 : MAX_N_NODES ** 2 + 2 * MAX_N_NODES] == 0).all()
    assert (pi[:, MAX_N_NODES ** 2 + 2 * MAX_N_NODES + 4 : MAX_N_NODES ** 2 + 3 * MAX_N_NODES] == 0).all()
    assert (pi[:, MAX_N_NODES ** 2 + 3 * MAX_N_NODES + 4 : MAX_N_NODES ** 2 + 4 * MAX_N_NODES] == 0).all()

    pi, value = me(features2)
    assert pi[0, 0] != 0
    assert pi[0, 2 * MAX_N_NODES + 2] != 0
    pi[0, 0] = 0
    pi[0, 2 * MAX_N_NODES + 2] = 0
    assert pi[0, MAX_N_NODES ** 2] != 0
    assert pi[0, MAX_N_NODES ** 2 + 2 * MAX_N_NODES + 2] != 0
    pi[0, MAX_N_NODES ** 2] = 0
    pi[0, MAX_N_NODES ** 2 + 2 * MAX_N_NODES + 2] = 0
    assert (pi[0] == 0).all()
    assert pi[1, 3 * MAX_N_NODES + 3] != 0
    pi[1, 3 * MAX_N_NODES + 3] = 0
    assert pi[1, MAX_N_NODES ** 2 + 3 * MAX_N_NODES + 3] != 0
    pi[1, MAX_N_NODES ** 2 + 3 * MAX_N_NODES + 3] = 0
    assert (pi == 0).all()
