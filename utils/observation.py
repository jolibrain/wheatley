from torch_geometric.data import Data, DataLoader


class Observation:
    def __init__(self, obs):
        if isinstance(obs["n_nodes"], int):
            self.n_nodes = obs["n_nodes"]
        else:
            n_dims = len(list(obs["n_nodes"].shape))
            if n_dims == 3:
                n_nodes = (obs["n_nodes"][0][0] == 1).nonzero(as_tuple=True)[0]
            else:
                n_nodes = (obs["n_nodes"][0] == 1).nonzero(as_tuple=True)[0]
            self.n_nodes = n_nodes.item()

        self.features = obs["features"][
            :, 0 : self.n_nodes, :
        ]  # delete useless features
        self.edge_index = obs["edge_index"].long()

    def get_batch_size(self):
        return self.features.shape[0]

    def get_n_nodes(self):
        return self.n_nodes

    def get_graph(self):
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
