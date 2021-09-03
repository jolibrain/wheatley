import torch

from config import MAX_N_NODES, MAX_N_EDGES


class EnvObservation:
    def __init__(
        self, n_jobs, n_machines, n_nodes, n_edges, features, edge_index, mask
    ):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.features = features
        self.edge_index = edge_index
        self.mask = mask
        assert self.n_nodes == self.n_jobs * self.n_machines
        assert self.n_nodes == self.features.shape[0]
        assert self.n_edges == self.edge_index.shape[1]
        assert self.n_nodes ** 2 == self.mask.shape[0]

    def get_n_nodes(self):
        return self.n_nodes

    def get_n_edges(self):
        return self.n_edges

    @classmethod
    def from_torch_geometric(cls, n_jobs, n_machines, graph, mask):
        """
        This should only hanlde cpu tensors, since it is used on the env side.
        """
        if graph.x.is_cuda:
            raise Exception("Please provide a cpu observation")

        n_nodes = n_jobs * n_machines
        n_edges = graph.edge_index.shape[1]

        return cls(
            n_jobs,
            n_machines,
            n_nodes,
            n_edges,
            graph.x,
            graph.edge_index,
            mask,
        )

    def to_gym_observation(self):
        """
        This should only handle cpu tensors, since it is used on the env side.
        """
        if self.features.is_cuda:
            raise Exception("Please use this only on cpu observation")

        features = torch.zeros((MAX_N_NODES, self.features.shape[1]))
        features[0 : self.get_n_nodes(), :] = self.features
        edge_index = torch.zeros((2, MAX_N_EDGES))
        edge_index[:, 0 : self.get_n_edges()] = self.edge_index
        mask = torch.zeros(MAX_N_EDGES)
        mask[0 : self.get_n_edges()] = self.mask
        return {
            "n_jobs": self.n_jobs,
            "n_machines": self.n_machines,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "features": features.numpy(),
            "edge_index": edge_index.numpy(),
            "mask": mask.numpy(),
        }
