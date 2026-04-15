from abc import ABC, abstractmethod


class Graph(ABC):
    @abstractmethod
    def __init__(
        self,
        problem_edges,
        num_nodes,
        factored_rp,
        observe_conflicts_as_cliques,
        device,
    ):
        pass

    @abstractmethod
    def ndata(nodetype, featname):
        pass

    @abstractmethod
    def set_ndata(self, nodetype, featname, t):
        pass

    @abstractmethod
    def num_nodes(self, ntype):
        pass

    @abstractmethod
    def predecessors(self, nid):
        pass

    @abstractmethod
    def successors(self, nid):
        pass

    @abstractmethod
    def indeg(self, nid):
        pass

    @abstractmethod
    def in_degrees(self):
        pass

    @abstractmethod
    def out_degrees(self):
        pass

    @abstractmethod
    def set_global_data(self, featname, data):
        pass

    @abstractmethod
    def global_data(self, featname):
        pass

    @abstractmethod
    def edges(self, etype):
        pass

    @abstractmethod
    def remove_edges(self, eid, etype):
        pass

    @abstractmethod
    def add_edges(self, sources, destinations, data, etype):
        pass

    @abstractmethod
    def num_edges(self, etype):
        pass

    @abstractmethod
    def edata(self, etype, dataid):
        pass

    @classmethod
    @abstractmethod
    def batch(cls, graphlist, num_nodes, num_edges):
        pass

    @abstractmethod
    def save(self, fname):
        pass

    @classmethod
    @abstractmethod
    def load(cls, fname):
        pass
