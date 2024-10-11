from abc import ABC, abstractmethod
from .dgl_graph import DGLGraph
from .pyg_graph import PYGGraph
from torch_geometric.data import Batch


class GraphFactory:
    @classmethod
    def create_graph(
        cls,
        problem_edges,
        num_nodes,
        factored_rp,
        observe_conflicts_as_cliques,
        device,
        pyg=False,
    ):
        if not pyg:
            return DGLGraph(
                problem_edges,
                num_nodes,
                factored_rp,
                observe_conflicts_as_cliques,
                device,
            )
        else:
            return PYGGraph(
                problem_edges,
                num_nodes,
                factored_rp,
                observe_conflicts_as_cliques,
                device,
            )

    @classmethod
    def load(cls, fname, pyg=True):
        if pyg:
            return PYGGraph.load(fname)
        else:
            return DGLGraph.load(fname)

    @classmethod
    def deserialize(cls, bytearr, pyg=True):
        if pyg:
            return PYGGraph.deserialize(bytearr)
        else:
            raise RuntimeError("not implemented")
