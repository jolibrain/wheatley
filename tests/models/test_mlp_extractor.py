import torch

from models.mlp_extractor import MLPExtractor


def test_get_pairs_to_compute():
    me = MLPExtractor(
        add_boolean=False,
        mlp_act="tanh",
        device=torch.device("cpu"),
        input_dim_features_extractor=2,
        max_n_nodes=4,
        max_n_jobs=2,
        n_layers_features_extractor=4,
        hidden_dim_features_extractor=64,
        n_mlp_layers_actor=4,
        hidden_dim_actor=32,
        n_mlp_layers_critic=4,
        hidden_dim_critic=32,
    )
    graph_embedding = torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]])
    nodes_embedding = torch.tensor([[[10, 11], [12, 13]], [[14, 15], [16, 17]], [[18, 19], [20, 21]], [[22, 23], [24, 25]]])
    mask = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    pairs, indexes = me._get_pairs_to_compute(graph_embedding, nodes_embedding, mask)

    assert (
        pairs
        == torch.tensor(
            [
                [[0, 1, 12, 13], [0, 0, 0, 0]],
                [[2, 3, 16, 17], [0, 0, 0, 0]],
                [[4, 5, 18, 19], [0, 0, 0, 0]],
                [[6, 7, 22, 23], [6, 7, 24, 25]],
            ]
        )
    ).all()


def test_forward():
    # Without adding boolean
    me = MLPExtractor(
        add_boolean=False,
        mlp_act="tanh",
        device=torch.device("cpu"),
        input_dim_features_extractor=2,
        max_n_nodes=4,
        max_n_jobs=2,
        n_layers_features_extractor=4,
        hidden_dim_features_extractor=64,
        n_mlp_layers_actor=4,
        hidden_dim_actor=32,
        n_mlp_layers_critic=4,
        hidden_dim_critic=32,
    )
    features1 = torch.cat(
        [
            torch.rand(
                (1, 5, 2 + 64 * 4),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
            torch.tensor(
                [
                    [
                        [0],
                        [1],
                        [0],
                        [1],
                        [0],
                    ]
                ],
                device=torch.device("cpu"),
            ),
        ],
        axis=2,
    )
    pi, value = me(features1)
    assert list(value.shape) == [1, 1, 1]
    assert list(pi.shape) == [1, 4]
    # Check that the 0 corresponding to null nodes are in the right positions
    assert (pi[:, 0] != 0).all()
    assert (pi[:, 1] == 0).all()
    assert (pi[:, 2] != 0).all()
    assert (pi[:, 3] == 0).all()

    features2 = torch.tensor(
        [
            [
                [1 for i in range(2 + 64 * 4)] + [0],
                [2 for i in range(2 + 64 * 4)] + [1],
                [3 for i in range(2 + 64 * 4)] + [0],
                [4 for i in range(2 + 64 * 4)] + [1],
                [5 for i in range(2 + 64 * 4)] + [0],
            ],
            [
                [1 for i in range(2 + 64 * 4)] + [0],
                [2 for i in range(2 + 64 * 4)] + [0],
                [3 for i in range(2 + 64 * 4)] + [0],
                [4 for i in range(2 + 64 * 4)] + [0],
                [5 for i in range(2 + 64 * 4)] + [1],
            ],
        ],
        dtype=torch.float,
        device=torch.device("cpu"),
    )
    pi, value = me(features2)
    assert pi[0, 0] != 0
    assert pi[0, 2] != 0
    pi[0, 0] = 0
    pi[0, 2] = 0
    assert (pi[0] == 0).all()
    assert pi[1, 3] != 0
    pi[1, 3] = 0
    assert (pi == 0).all()

    # With adding a boolean
    me = MLPExtractor(
        add_boolean=True,
        mlp_act="tanh",
        device=torch.device("cpu"),
        input_dim_features_extractor=2,
        max_n_nodes=4,
        max_n_jobs=2,
        n_layers_features_extractor=4,
        hidden_dim_features_extractor=64,
        n_mlp_layers_actor=4,
        hidden_dim_actor=32,
        n_mlp_layers_critic=4,
        hidden_dim_critic=32,
    )
    pi, value = me(features1)
    assert list(pi.shape) == [1, 2 * 4]
    # Check that the 0 corresponding to null nodes are in the right positions
    assert (pi[:, 0] != 0).all()
    assert (pi[:, 1] == 0).all()
    assert (pi[:, 2] != 0).all()
    assert (pi[:, 3] == 0).all()
    assert (pi[:, 4] != 0).all()
    assert (pi[:, 5] == 0).all()
    assert (pi[:, 6] != 0).all()
    assert (pi[:, 7] == 0).all()

    pi, value = me(features2)
    assert pi[0, 0] != 0
    assert pi[0, 2] != 0
    assert pi[0, 4] != 0
    assert pi[0, 6] != 0
    pi[0, 0] = 0
    pi[0, 2] = 0
    pi[0, 4] = 0
    pi[0, 6] = 0
    assert (pi[0] == 0).all()
    assert pi[1, 3] != 0
    assert pi[1, 7] != 0
    pi[1, 3] = 0
    pi[1, 7] = 0
    assert (pi == 0).all()
