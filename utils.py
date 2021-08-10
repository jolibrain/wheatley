import gym
from torch_geometric.data import Data


def gym2graph(state):
    """
    Convert a dot in a tuple of box gym space into a graph
    """
    features = state[0]
    edge_index = state[1]
    return Data(x=features, edge_index=edge_index)


def graph2gym(graph):
    return (graph.x, graph.edge_index)
