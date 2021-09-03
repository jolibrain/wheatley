import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv

from models.mlp import MLP
from utils.utils import apply_mask

from config import (
    HIDDEN_DIM_FEATURES_EXTRACTOR,
    N_MLP_LAYERS_ACTOR,
    HIDDEN_DIM_ACTOR,
    N_MLP_LAYERS_CRITIC,
    HIDDEN_DIM_CRITIC,
    MAX_N_NODES,
    DEVICE,
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

    def forward(self, embedded_features):
        """
        Takes nodes_and_graph_embedding as input. This should be the output of the
        FeatureExtractor
        """
        # First decompose the features into mask, graph_embedding and nodes_embedding
        graph_embedding, nodes_embedding, mask = self._decompose_features(
            embedded_features
        )
        n_nodes = nodes_embedding.shape[1]
        batch_size = graph_embedding.shape[0]

        # Then compute actor and critic
        value = self.critic(graph_embedding)

        possible_s_a_pairs = self._compute_possible_s_a_pairs(
            graph_embedding, nodes_embedding
        )

        # Apply a mask
        pairs_to_compute, indexes = apply_mask(possible_s_a_pairs, mask)

        # Compute the probabilities
        probabilities = self.actor(pairs_to_compute)
        pi = F.softmax(probabilities, dim=1)

        # And reshape pi in ordrer to have every value corresponding to its edge index
        shaped_pi = torch.zeros((batch_size, n_nodes * n_nodes), device=DEVICE)
        for i in range(batch_size):
            shaped_pi[i][indexes[i]] = pi[i].reshape(pi.shape[1])

        # The final pi must include all actions (even those that are not applicable
        # because of the size of the graph). So we convert the flattened pi to a square
        # matrix of size (n_nodes, n_nodes). We then complete the matrix to get a square
        # matrix of size (MAX_N_NODES, MAX_N_NODES) with 0, and we reflatten it.
        shaped_pi = shaped_pi.reshape(batch_size, n_nodes, n_nodes)
        filled_pi = torch.zeros(
            (batch_size, MAX_N_NODES, MAX_N_NODES), device=DEVICE
        )
        filled_pi[:, 0:n_nodes, 0:n_nodes] = shaped_pi
        filled_pi = filled_pi.reshape(batch_size, MAX_N_NODES * MAX_N_NODES)
        return filled_pi, value

    def _compute_possible_s_a_pairs(self, graph_embedding, nodes_embedding):
        # We create 3 tensors representing state, node 1 and node 2
        # and then stack them together to get all state action pairs
        n_nodes = nodes_embedding.shape[1]

        states = graph_embedding.repeat(1, n_nodes * n_nodes, 1)
        nodes1 = nodes_embedding.repeat(1, n_nodes, 1)
        nodes2 = nodes_embedding.repeat_interleave(n_nodes, dim=1)

        s_a_pairs = torch.cat([states, nodes1, nodes2], dim=2)
        return s_a_pairs

    def _decompose_features(self, embedded_features):
        graph_and_nodes_embedding, extended_mask = torch.split(
            embedded_features,
            [
                HIDDEN_DIM_FEATURES_EXTRACTOR,
                embedded_features.shape[2] - HIDDEN_DIM_FEATURES_EXTRACTOR,
            ],
            dim=2,
        )
        graph_embedding, nodes_embedding = torch.split(
            graph_and_nodes_embedding,
            [1, graph_and_nodes_embedding.shape[1] - 1],
            dim=1,
        )
        batch_size = graph_embedding.shape[0]
        mask = extended_mask[:, 1:, :].reshape(batch_size, -1)
        return graph_embedding, nodes_embedding, mask
