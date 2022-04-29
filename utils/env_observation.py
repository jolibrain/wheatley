import torch


class EnvObservation:
    def __init__(self, n_jobs, n_machines, features, edge_index, max_n_jobs, max_n_machines):
        """
        This should only hanlde cpu tensors, since it is used on the env side.
        """
        if features.is_cuda:
            raise Exception("Please provide a cpu observation")

        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.max_n_jobs = max_n_jobs
        self.max_n_machines = max_n_machines
        self.max_n_nodes = self.max_n_jobs * self.max_n_machines
        self.max_n_edges = self.max_n_nodes ** 2
        self.n_nodes = n_jobs * n_machines
        self.n_edges = edge_index.shape[1]
        self.features = features
        self.edge_index = edge_index
        assert self.n_nodes == self.features.shape[0]

    def get_n_nodes(self):
        return self.n_nodes

    def get_n_edges(self):
        return self.n_edges

    def to_gym_observation(self):
        """
        This should only handle cpu tensors, since it is used on the env side.
        """
        if self.features.is_cuda:
            raise Exception("Please use this only on cpu observation")

        features = torch.zeros((self.max_n_nodes, self.features.shape[1]))
        features[0 : self.get_n_nodes(), :] = self.features
        edge_index = torch.zeros((2, self.max_n_edges))
        edge_index[:, 0 : self.get_n_edges()] = self.edge_index
        return {
            "n_jobs": self.n_jobs,
            "n_machines": self.n_machines,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "features": features.numpy(),
            "edge_index": edge_index.numpy().astype('int64'),
        }
