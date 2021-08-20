import numpy as np
from torch_geometric.data import Data, DataLoader

from config import MAX_N_NODES, MAX_N_EDGES


class Observation:
    def __init__(self, n_nodes, features, edge_index):
        self.n_nodes = n_nodes
        self.features = features
        self.edge_index = edge_index

    @classmethod
    def from_gym_observation(cls, gym_observation):
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

        features = gym_observation["features"][
            :, 0:n_nodes, :
        ]  # delete useless features
        edge_index = gym_observation["edge_index"].long()
        return cls(n_nodes, features, edge_index)

    @classmethod
    def from_torch_geometric(cls, graph):
        return cls(
            graph.x.shape[0],
            graph.x.unsqueeze(0),
            graph.edge_index.unsqueeze(0),
        )

    def get_batch_size(self):
        return self.features.shape[0]

    def get_n_nodes(self):
        return self.n_nodes

    def to_torch_geometric(self):
        """
        Returns the batched graph associated with the observation
        """
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
        n_edges = self.edge_index.shape[2]
        filled_features = np.zeros((MAX_N_NODES, 3))
        filled_edge_index = np.zeros((2, MAX_N_EDGES))
        filled_features[0 : self.n_nodes, :] = self.features.numpy().reshape(
            self.n_nodes, 3
        )
        filled_edge_index[:, 0:n_edges] = self.edge_index.numpy().reshape(
            2, n_edges
        )
        return {
            "n_nodes": self.n_nodes,
            "features": filled_features,
            "edge_index": filled_edge_index,
        }

    def drop_node_ids(self):
        node_ids = self.features[:, :, 0]
        self.features = self.features[:, :, 1:]
        return node_ids
