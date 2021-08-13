import torch
from torch_geometric.data import Data, DataLoader

from models.actor_critic import ActorCritic

from config import DEVICE, HIDDEN_DIM_FEATURES_EXTRACTOR, MAX_N_NODES


def test_compute_possible_s_a_pairs():
    ac = ActorCritic()
    graph_embedding = torch.tensor([[[0, 1]], [[2, 3]]])
    nodes_embedding = torch.tensor([[[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    assert torch.eq(
        ac.compute_possible_s_a_pairs(graph_embedding, nodes_embedding),
        torch.tensor(
            [
                [
                    [0, 1, 4, 5, 4, 5],
                    [0, 1, 6, 7, 4, 5],
                    [0, 1, 4, 5, 6, 7],
                    [0, 1, 6, 7, 6, 7],
                ],
                [
                    [2, 3, 8, 9, 8, 9],
                    [2, 3, 10, 11, 8, 9],
                    [2, 3, 8, 9, 10, 11],
                    [2, 3, 10, 11, 10, 11],
                ],
            ]
        ),
    ).all()


def test_forward():
    ac = ActorCritic()
    features = torch.rand(
        (1, 5, HIDDEN_DIM_FEATURES_EXTRACTOR), dtype=torch.float32, device=DEVICE
    )
    pi, value = ac(features)
    assert list(value.shape) == [1, 1, 1]
    assert list(pi.shape) == [1, MAX_N_NODES ** 2]
    # Check that the 0 corresponding to null nodes are in the right positions
    assert (pi[:, 4:MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + MAX_N_NODES : 2 * MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + 2 * MAX_N_NODES : 3 * MAX_N_NODES] == 0).all()
    assert (pi[:, 4 + 3 * MAX_N_NODES : 4 * MAX_N_NODES] == 0).all()
