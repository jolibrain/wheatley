import torch
from torch_geometric.data import Data, DataLoader

from utils.state import State

from config import MAX_N_NODES, DEVICE


def test_from_graph():
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 3, 2], [1, 0, 3, 2, 2, 3]],
        dtype=torch.int64,
        device=DEVICE,
    )
    features = torch.rand((4, 2), dtype=torch.float32, device=DEVICE)
    state = State.from_graph(Data(x=features, edge_index=edge_index))
    assert state.n_nodes == 4
    assert torch.eq(state.features, features).all()
    assert torch.eq(state.edge_index, edge_index).all()


def test_from_batch_graph():
    batched_edge_index = torch.tensor(
        [
            [[0, 1, 2, 3, 3, 2], [1, 0, 3, 2, 2, 3]],
            [[0, 3, 2, 1, 2, 0], [2, 1, 2, 3, 3, 0]],
        ],
        dtype=torch.int64,
        device=DEVICE,
    )
    features = torch.rand((2, 4, 2), dtype=torch.float32, device=DEVICE)
    loader = DataLoader(
        [Data(x=features[i], edge_index=batched_edge_index[i]) for i in range(2)],
        batch_size=2,
    )
    graph = next(iter(loader))
    state = State.from_batch_graph(graph, batched_edge_index, 2)
    assert state.n_nodes == 4
    assert torch.eq(state.features, features).all()
    assert torch.eq(state.edge_index, batched_edge_index).all()


def test_from_observation():
    n_nodes1 = 4
    n_nodes2 = torch.zeros((1, MAX_N_NODES), device=DEVICE)
    n_nodes2[:, 4] = 1
    n_nodes3 = torch.zeros((4, 1, MAX_N_NODES), device=DEVICE)
    n_nodes3[:, :, 4] = 1
    features1 = torch.rand((1, MAX_N_NODES, 2), dtype=torch.float32, device=DEVICE)
    features1[:, 4:MAX_N_NODES, :] = 0
    real_features1 = features1[:, 0:4, :]
    features2 = torch.rand((4, MAX_N_NODES, 2), dtype=torch.float32, device=DEVICE)
    features2[:, 4:MAX_N_NODES, :] = 0
    real_features2 = features2[0:4, 0:4, :]
    edge_index1 = torch.zeros((1, 2, MAX_N_NODES ** 2))
    edge_index1[:, :, 0:5] = torch.tensor([[[0, 1, 3, 2, 1], [1, 3, 2, 2, 0]]])
    edge_index2 = torch.zeros((4, 2, MAX_N_NODES ** 2))
    edge_index2[:, :, 0:5] = torch.tensor([[[0, 1, 3, 2, 1], [1, 3, 2, 2, 0]]])

    state1 = State.from_observation(
        {"n_nodes": n_nodes1, "features": features1, "edge_index": edge_index1}
    )
    state2 = State.from_observation(
        {"n_nodes": n_nodes2, "features": features1, "edge_index": edge_index1}
    )
    state3 = State.from_observation(
        {"n_nodes": n_nodes3, "features": features2, "edge_index": edge_index2}
    )

    assert state1.n_nodes == 4
    assert state2.n_nodes == 4
    assert state3.n_nodes == 4
    assert torch.eq(state1.features, real_features1).all()
    assert torch.eq(state3.features, real_features2).all()
    assert torch.eq(state1.edge_index, edge_index1).all()
    assert torch.eq(state3.edge_index, edge_index2).all()
