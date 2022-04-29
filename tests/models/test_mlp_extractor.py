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
    pairs, indexes = me._get_pairs_to_compute(graph_embedding, nodes_embedding)

    assert (
        pairs
        == torch.tensor(
            [
                [[0, 1, 10, 11], [0, 1, 12, 13]],
                [[2, 3, 14, 15], [2, 3, 16, 17]],
                [[4, 5, 18, 19], [4, 5, 20, 21]],
                [[6, 7, 22, 23], [6, 7, 24, 25]]
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
        ],
        axis=2,
    )
    pi, value = me(features1)
    assert list(value.shape) == [1, 1, 1]
    assert list(pi.shape) == [1, 4]

    features2 = torch.tensor(
        [
            [
                [1 for i in range(2 + 64 * 4)],
                [2 for i in range(2 + 64 * 4)],
                [3 for i in range(2 + 64 * 4)],
                [4 for i in range(2 + 64 * 4)],
                [5 for i in range(2 + 64 * 4)],
            ],
            [
                [1 for i in range(2 + 64 * 4)],
                [2 for i in range(2 + 64 * 4)],
                [3 for i in range(2 + 64 * 4)],
                [4 for i in range(2 + 64 * 4)],
                [5 for i in range(2 + 64 * 4)],
            ],
        ],
        dtype=torch.float,
        device=torch.device("cpu"),
    )
    pi, value = me(features2)

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

    pi, value = me(features2)
