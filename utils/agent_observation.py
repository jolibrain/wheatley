import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class AgentObservation:
    def __init__(self, n_jobs, n_machines, n_nodes, n_edges, features, edge_index, mask):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.features = features
        self.edge_index = edge_index
        self.mask = mask
        assert self.n_nodes == self.n_jobs * self.n_machines
        assert self.n_nodes == self.features.shape[1]
        assert self.n_edges == self.edge_index.shape[2]
        assert self.n_nodes ** 2 == self.mask.shape[1]

    def get_batch_size(self):
        return self.features.shape[0]

    def get_n_nodes(self):
        return self.n_nodes

    def get_mask(self):
        return self.mask

    @classmethod
    def from_gym_observation(cls, gym_observation):
        """
        This should only hanlde cuda tensors, since it is used on the agent side
        (if and only if cuda is available on the machine).
        """
        # if torch.cuda.is_available():
        #     if not gym_observation["features"].is_cuda:
        #         raise Exception("Please provide a cuda observation")

        if isinstance(gym_observation["n_jobs"], int):
            n_jobs = gym_observation["n_jobs"]
            n_machines = gym_observation["n_machines"]
            n_nodes = gym_observation["n_nodes"]
            n_edges = gym_observation["n_edges"]
        else:
            n_dims = len(list(gym_observation["n_jobs"].shape))
            if n_dims == 3:
                n_jobs = (gym_observation["n_jobs"][0][0] == 1).nonzero(as_tuple=True)[0]
                n_machines = (gym_observation["n_machines"][0][0] == 1).nonzero(as_tuple=True)[0]
                n_nodes = (gym_observation["n_nodes"][0][0] == 1).nonzero(as_tuple=True)[0]
                n_edges = (gym_observation["n_edges"][0][0] == 1).nonzero(as_tuple=True)[0]
            else:
                n_jobs = (gym_observation["n_jobs"][0] == 1).nonzero(as_tuple=True)[0]
                n_machines = (gym_observation["n_machines"][0] == 1).nonzero(as_tuple=True)[0]
                n_nodes = (gym_observation["n_nodes"][0] == 1).nonzero(as_tuple=True)[0]
                n_edges = (gym_observation["n_edges"][0] == 1).nonzero(as_tuple=True)[0]

            n_jobs = n_jobs.item()
            n_machines = n_machines.item()
            n_nodes = n_nodes.item()
            n_edges = n_edges.item()

        features = gym_observation["features"][:, 0:n_nodes, :]
        edge_index = gym_observation["edge_index"][:, :, 0:n_edges].long()
        mask = gym_observation["mask"][:, 0 : n_nodes ** 2]
        return cls(n_jobs, n_machines, n_nodes, n_edges, features, edge_index, mask)

    def to_torch_geometric(self):
        """
        Returns the batched graph associated with the observation.
        This should only hanlde cuda tensors, since it is used on the agent side
        (if and only if cuda is available on the machine).
        """
        # if torch.cuda.is_available():
        #     if not self.features.is_cuda:
        #         raise Exception("Please use this only on cuda observation")

        loader = DataLoader(
            [Data(self.features[i], self.edge_index[i]) for i in range(self.get_batch_size())],
            batch_size=self.get_batch_size(),
        )
        graph = next(iter(loader))
        return graph
