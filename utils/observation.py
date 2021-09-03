import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from config import MAX_N_NODES, MAX_N_EDGES


class Observation:
    """
    This class is used to describe an observation in all the steps of the training
    phase. So it should be able to represent the prepared observation, that will be
    passed to the Env class; but it should also be able to represent the observation
    the model is going to get after the gym env (and the manipulations due to PPO).
    Because the environnement pass the observation from numpy to torch and from
    CPU to GPU (if possible), the different functions handle different cases,
    depending on where they are going to be used.
    """

    def __init__(self, n_jobs, n_machines, features, edge_index, mask):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.features = features
        self.edge_index = edge_index
        self.mask = mask

    @classmethod
    def from_gym_observation(cls, gym_observation):
        """
        This should only hanlde cuda tensors, since it is used on the agent side
        (if and only if cuda is available on the machine).
        """
        if torch.cuda.is_available():
            if not gym_observation["features"].is_cuda:
                raise Exception("Please provide a cuda observation")

        if isinstance(gym_observation["n_nodes"], int):
            n_nodes = gym_observation["n_nodes"]
        else:
            n_dims = len(list(gym_observation["n_nodes"].shape))
            if n_dims == 3:
                n_nodes = (gym_observation["n_nodes"][0][0] == 1).nonzero(
                    as_tuple=True
                )[0]
            else:
                n_nodes = (gym_observation["n_nodes"][0] == 1).nonzero(
                    as_tuple=True
                )[0]
            n_nodes = n_nodes.item()

        features = gym_observation["features"]
        edge_index = gym_observation["edge_index"].long()
        mask = gym_observation["mask"]
        return cls(n_nodes, features, edge_index, mask)

    @classmethod
    def from_torch_geometric(cls, graph, mask):
        """
        This should only hanlde cpu tensors, since it is used on the env side.
        """
        if graph.x.is_cuda:
            raise Exception("Please provide a cpu observation")
        
        # We have to reorder features, since the networx -> torch_geometric
        # shuffles the nodes
        n_nodes = graph.x.shape[0]
        node_ids = graph.x[:, 0]
        features = torch.zeros((MAX_N_NODES, graph.x[:, 1:].shape[1]))
        features[node_ids] = graph.x[:, 1:]

        edge_index = node_ids[graph.edge_index]
        
        return cls(
            n_nodes,
            features.unsqueeze(0)
            edge_index.unsqueeze(0),
            mask.unsqueeze(0),
        )

    def get_batch_size(self):
        return self.features.shape[0]

    def get_n_nodes(self):
        return self.n_nodes

    def get_mask(self):
        return self.mask

    def to_torch_geometric(self):
        """
        Returns the batched graph associated with the observation.
        This should only hanlde cuda tensors, since it is used on the agent side
        (if and only if cuda is available on the machine).
        """
        if torch.cuda.is_available():
            if not self.features.is_cuda:
                raise Exception("Please use this only on cuda observation")

        loader = DataLoader(
            [
                Data(self.features[i], self.edge_index[i])
                for i in range(self.get_batch_size())
            ],
            batch_size=self.get_batch_size(),
        )
        graph = next(iter(loader))
        return graph

    def to_gym_observation(self):
        """
        This should only handle cpu tensors, since it is used on the env side.
        """
        if self.features.is_cuda:
            raise Exception("Please use this only on cpu observation")

        return {
            "n_nodes": self.n_nodes,
            "features": self.features.numpy(),
            "edge_index": self.edge_index.numpy(),
            "mask": self.mask.numpy(),
        }

