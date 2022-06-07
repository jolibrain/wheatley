import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import time


class AgentObservation:
    def __init__(self, graph):
        self.graph = graph

    def get_batch_size(self):
        return self.graph.num_graphs

    def get_n_nodes(self):
        return int(self.graph.num_nodes / self.graph.num_graphs)

    @classmethod
    def from_gym_observation(cls, gym_observation):

        if isinstance(gym_observation["n_jobs"], int):
            # n_jobs = gym_observation["n_jobs"]
            # n_machines = gym_observation["n_machines"]
            n_nodes = gym_observation["n_nodes"]
            n_edges = gym_observation["n_edges"]
        else:
            n_dims = gym_observation["n_jobs"].dim()
            if n_dims == 3:
                # n_jobs = torch.argmax(gym_observation["n_jobs"][0][0])
                # n_machines = torch.argmax(gym_observation["n_machines"][0][0])
                n_nodes = torch.argmax(gym_observation["n_nodes"][0][0])
                n_edges = torch.argmax(gym_observation["n_edges"][0][0])
            else:
                # n_jobs = torch.argmax(gym_observation["n_jobs"][0])
                # n_machines = torch.argmax(gym_observation["n_machines"][0])
                n_nodes = torch.argmax(gym_observation["n_nodes"][0])
                n_edges = torch.argmax(gym_observation["n_edges"][0])

        features = gym_observation["features"][:, :n_nodes, :]
        edge_index = gym_observation["edge_index"][:, :, :n_edges].long()
        # graph = Batch.from_data_list([Data(features[i], edge_index[i]) for i in range(features.shape[0])])
        # collating is much faster on cpu due to transfer of incs
        orig_device = features.device
        fcpu = features.to("cpu")
        eicpu = edge_index.to("cpu")

        graph = Batch.from_data_list([Data(fcpu[i], eicpu[i]) for i in range(fcpu.shape[0])])
        return cls(graph.to(orig_device))

    def to_torch_geometric(self):
        """
        Returns the batched graph associated with the observation.
        """
        return self.graph
