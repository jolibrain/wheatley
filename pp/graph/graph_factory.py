from abc import ABC, abstractmethod

# from .dgl_graph import DGLGraph
from .pyg_graph import PYGGraph
from torch_geometric.data import Batch


class GraphFactory:
    @classmethod
    def create_graph(
        cls,
        problem_edges,
        num_nodes,
        device=None,
    ):
        return PYGGraph(
            problem_edges,
            num_nodes,
            device,
        )

    @classmethod
    def load(cls, fname, pyg=True):
        return PYGGraph.load(fname)

    @classmethod
    def deserialize(cls, bytearr, pyg=True):
        return PYGGraph.deserialize(bytearr)
