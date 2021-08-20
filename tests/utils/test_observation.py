import torch

from utils.observation import Observation


def test_from_gym_observation():
    obs = Observation.from_gym_observation(
        {
            "n_nodes": torch.tensor([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]]),
            "features": torch.tensor(
                [
                    [
                        [0, 1, 4],
                        [2, 1, 7],
                        [1, 1, 1],
                        [3, 0, 6],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ]
                ]
            ),
            "edge_index": torch.tensor(
                [[[0, 1, 2, 3], [1, 2, 3, 0]]], dtype=torch.int64
            ),
        }
    )
    assert obs.n_nodes == 4
    assert list(obs.features.shape) == [1, 4, 3]
    assert list(obs.edge_index.shape) == [1, 2, 4]


def test_from_torch_geometric(graph):
    obs = Observation.from_torch_geometric(graph)
    assert obs.n_nodes == 4
    assert list(obs.features.shape) == [1, 4, 3]
    assert list(obs.edge_index.shape) == [1, 2, 4]


def test_to_torch_geometric(observation):
    graph = observation.to_torch_geometric()
    assert list(graph.x.shape) == [18, 3]
    assert list(graph.edge_index.shape) == [2, 16]


def test_to_gym_observation(observation):
    observation.features = observation.features[1:, :, :].reshape(1, 9, 3)
    observation.edge_index = observation.edge_index[1, :, :].reshape(1, 2, 8)
    gym_obs = observation.to_gym_observation()

    assert gym_obs["n_nodes"] == 9
    assert list(gym_obs["features"].shape) == [100, 3]
    assert list(gym_obs["edge_index"].shape) == [2, 10000]
    assert (gym_obs["features"][9:, :] == 0).all()
    assert (gym_obs["edge_index"][:, 8:] == 0).all()


def test_drop_nodes_ids(observation):
    node_ids = observation.drop_node_ids()
    assert list(observation.features.shape) == [2, 9, 2]
    assert list(node_ids.shape) == [2, 9]
    assert (observation.features[:, :, 0] <= 1).all()
    assert (observation.features[:, :, 1] >= 0).all()
    assert torch.eq(
        node_ids,
        torch.tensor(
            [0, 1, 3, 2, 4, 7, 8, 5, 6, 0, 1, 3, 2, 4, 7, 8, 5, 6]
        ).reshape(2, 9),
    ).all()
