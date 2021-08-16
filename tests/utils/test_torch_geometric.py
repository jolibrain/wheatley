import torch
from torch_geometric.data import Data

from utils.torch_geometric_utils import find_edge, add_edge, remove_edge


def test_find_edge(graph):
    assert find_edge(graph, (2, 3)) == 2
    assert find_edge(graph, (1, 3)) is None


def test_add_edge(graph):
    assert find_edge(graph, (1, 3)) is None
    graph = add_edge(graph, (1, 3))
    assert find_edge(graph, (1, 3)) == 4


def test_remove_edge(graph):
    assert find_edge(graph, (2, 3)) == 2
    graph = remove_edge(graph, (2, 3))
    assert find_edge(graph, (2, 3)) is None
    assert find_edge(graph, (3, 0)) == 2
    graph = remove_edge(graph, (1, 3))
    assert list(graph.edge_index.shape) == [2, 3]
