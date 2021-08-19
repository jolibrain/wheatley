import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv

from models.mlp import MLP

from config import (
    HIDDEN_DIM_FEATURES_EXTRACTOR,
    N_MLP_LAYERS_ACTOR,
    HIDDEN_DIM_ACTOR,
    N_MLP_LAYERS_CRITIC,
    HIDDEN_DIM_CRITIC,
    DEVICE,
    MAX_N_NODES,
)


class MLPExtractor(nn.Module):
    def __init__(self):
        super(MLPExtractor, self).__init__()

        # This is necessary because of stable_baselines3 API
        self.latent_dim_pi = MAX_N_NODES ** 2
        self.latent_dim_vf = 1

        self.actor = MLP(
            n_layers=N_MLP_LAYERS_ACTOR,
            input_dim=HIDDEN_DIM_FEATURES_EXTRACTOR * 3,
            hidden_dim=HIDDEN_DIM_ACTOR,
            output_dim=1,
            batch_norm=False,
            device=DEVICE,
        )
        self.critic = MLP(
            n_layers=N_MLP_LAYERS_CRITIC,
            input_dim=HIDDEN_DIM_FEATURES_EXTRACTOR,
            hidden_dim=HIDDEN_DIM_CRITIC,
            output_dim=1,
            batch_norm=False,
            device=DEVICE,
        )

    def forward(self, nodes_and_graph_embedding):
        """
        Takes nodes_and_graph_embedding as input. This should be the output of the
        FeatureExtractor
        """
        graph_embedding, nodes_embedding = torch.split(
            nodes_and_graph_embedding,
            [1, nodes_and_graph_embedding.shape[1] - 1],
            dim=1,
        )
        batch_size = graph_embedding.shape[0]
        n_nodes = nodes_embedding.shape[1]

        value = self.critic(graph_embedding)

        possible_s_a_pairs = self.compute_possible_s_a_pairs(
            graph_embedding, nodes_embedding
        )
        probabilities = self.actor(possible_s_a_pairs)
        pi = F.softmax(probabilities, dim=1)

        # The final pi must include all actions (even those that are not applicable
        # because of the size of the graph). So we convert the flattened pi to a square
        # matrix of size (n_nodes, n_nodes). We then complete the matrix to get a square
        # matrix of size (MAX_N_NODES, MAX_N_NODES) with 0, and we reflatten it.
        pi = pi.reshape(
            batch_size, n_nodes * n_nodes
        )  # Remove the dim 2 of size 1
        pi = pi.reshape(batch_size, n_nodes, n_nodes)
        filled_pi = torch.zeros(
            batch_size, MAX_N_NODES, MAX_N_NODES, device=DEVICE
        )
        filled_pi[:, 0:n_nodes, 0:n_nodes] = pi
        filled_pi = filled_pi.reshape(batch_size, MAX_N_NODES * MAX_N_NODES)
        return filled_pi, value

    def compute_possible_s_a_pairs(self, graph_embedding, nodes_embedding):
        # We create 3 tensors representing state, node 1 and node 2
        # and then stack them together to get all state action pairs
        n_nodes = nodes_embedding.shape[1]

        states = graph_embedding.repeat(1, n_nodes * n_nodes, 1)
        nodes1 = nodes_embedding.repeat(1, n_nodes, 1)
        nodes2 = nodes_embedding.repeat_interleave(n_nodes, dim=1)

        s_a_pairs = torch.cat([states, nodes1, nodes2], dim=2)
        return s_a_pairs
