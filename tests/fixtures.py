from pytest import fixture
import torch
from torch_geometric.data import Data


@fixture
def graph():
    graph = Data(
        x=torch.rand(4, 2),
        edge_index=torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.int64
        ),
    )
    return graph
