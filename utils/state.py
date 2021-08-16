import torch
from torch_geometric.data import Data, DataLoader

from config import MAX_N_NODES, MAX_N_EDGES


class State:
    """
    This class is supposed to handle batched states, under the form of batched graph as
    explained in https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html,
    or under the form of batched observations from vectorized environments, as in
    https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    """

    def __init__(self, n_nodes, features, edge_index):
        self.n_nodes = n_nodes
        self.features = features
        self.edge_index = edge_index

    @classmethod
    def from_graph(cls, graph):
        """
        This function is supposed to build a state from a single graph (not a big one
        which is the minibatching of several different graphs)
        """
        features = torch.unsqueeze(graph.x, 0)
        n_nodes = features.shape[1]
        edge_index = torch.unsqueeze(graph.edge_index, 0)
        return cls(n_nodes=n_nodes, features=features, edge_index=edge_index)

    @classmethod
    def from_batch_graph(cls, graph, batched_edge_index, batch_size):
        """
        Here we need the batched_edge_index (the original edge_index, before mixing
        graphs together as explained in the doc of torch_geometric), since we can't
        reconstruct it from the edge_index post batching
        """
        n_nodes = int(graph.x.shape[0] / batch_size)
        return cls(
            n_nodes=n_nodes,
            features=graph.x.reshape(batch_size, n_nodes, -1),
            edge_index=batched_edge_index,
        )

    @classmethod
    def from_observation(cls, observation):
        if isinstance(observation["n_nodes"], int):
            n_nodes = observation["n_nodes"]
        else:
            n_dims = len(list(observation["n_nodes"].shape))
            if n_dims == 3:
                batched = True
            elif n_dims == 2:
                batched = False
            else:
                raise Exception(
                    "The 'n_nodes' value should be a 2 or 3-dimensional"
                    "tensor or an integer"
                )

            # First get n_nodes. We have to deal cases differently if batched or not,
            # because the shape of the vector is different. Then, we take care of passing
            # from 1-hot encoding to regular numbers
            if batched:
                n_nodes = (observation["n_nodes"][0][0] == 1).nonzero(
                    as_tuple=True
                )[0]
            else:
                n_nodes = (observation["n_nodes"][0] == 1).nonzero(
                    as_tuple=True
                )[0]
            n_nodes = n_nodes.item()

        return cls(
            n_nodes=n_nodes,
            features=observation["features"][
                :, 0:n_nodes, :
            ],  # delete useless features
            edge_index=observation["edge_index"].long(),
        )

    def to_graph(self):
        data_list = [
            Data(x=self.features[i], edge_index=self.edge_index[i])
            for i in range(self.get_batch_size())
        ]
        loader = DataLoader(data_list, batch_size=self.get_batch_size())
        graph = next(iter(loader))
        return graph

    def to_observation(self):
        """
        This function should not be used if we handle batched states. So we are going to
        suppose that batch_size is 1 here. If not, we throw an error.
        """
        if self.get_batch_size() > 1:
            raise Exception(
                "The to_observation function should not be used with "
                "batched state"
            )
        filled_features = torch.zeros(MAX_N_NODES, 2)
        filled_features[0 : self.n_nodes, :] = self.features.reshape(
            self.n_nodes, -1
        )
        filled_edge_index = torch.zeros(2, MAX_N_EDGES)
        filled_edge_index[
            :, 0 : self.edge_index.shape[2]
        ] = self.edge_index.reshape(2, -1)
        observation = {
            "features": filled_features.numpy(),
            "edge_index": filled_edge_index.numpy(),
            "n_nodes": self.n_nodes,
        }
        return observation

    def get_batch_size(self):
        return self.features.shape[0]

    def get_n_nodes(self):
        return self.n_nodes
