import torch

from env.state import State


def test_to_torch_geometric(state):
    graph = state.to_torch_geometric()
    node_ids = graph.x[:, 0]
    node_ids = node_ids.squeeze()
    arg_sorted_ids = torch.argsort(node_ids)
    features = graph.x[arg_sorted_ids]
    assert torch.eq(
        features,
        torch.tensor(
            [
                [0, 0, 1],
                [1, 0, 6],
                [2, 0, 16],
                [3, 0, 23],
                [4, 0, 31],
                [5, 0, 5],
                [6, 0, 11],
                [7, 0, 14],
                [8, 0, 17],
                [9, 0, 21],
                [10, 0, 4],
                [11, 0, 8],
                [12, 0, 12],
                [13, 0, 16],
                [14, 0, 20],
                [15, 0, 5],
                [16, 0, 11],
                [17, 0, 18],
                [18, 0, 26],
                [19, 0, 35],
                [20, 0, 9],
                [21, 0, 17],
                [22, 0, 24],
                [23, 0, 30],
                [24, 0, 35],
            ]
        ),
    ).all()
