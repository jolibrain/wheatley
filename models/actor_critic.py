import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv

from models.mlp import MLP

from config import (
    N_MLP_LAYERS_FEATURE_EXTRACTOR,
    N_LAYERS_FEATURE_EXTRACTOR,
    INPUT_DIM_FEATURE_EXTRACTOR,
    HIDDEN_DIM_FEATURE_EXTRACTOR,
    N_MLP_LAYERS_ACTOR,
    HIDDEN_DIM_ACTOR,
    N_MLP_LAYERS_CRITIC,
    HIDDEN_DIM_CRITIC,
    DEVICE,
    MAX_N_NODES,
)


class ActorCritic(nn.Module):
    def __init__(
        self,
    ):
        super(ActorCritic, self).__init__()

        self.hidden_dim_feature_extractor = HIDDEN_DIM_FEATURE_EXTRACTOR

        # This is necessary because of stable_baselines3 API
        self.latent_dim_pi = MAX_N_NODES ** 2
        self.latent_dim_vf = 1

        self.actor = MLP(
            n_layers=N_MLP_LAYERS_ACTOR,
            input_dim=HIDDEN_DIM_FEATURE_EXTRACTOR * 3,
            hidden_dim=HIDDEN_DIM_ACTOR,
            output_dim=1,
            batch_norm=False,
            device=DEVICE,
        )
        self.critic = MLP(
            n_layers=N_MLP_LAYERS_CRITIC,
            input_dim=HIDDEN_DIM_FEATURE_EXTRACTOR,
            hidden_dim=HIDDEN_DIM_CRITIC,
            output_dim=1,
            batch_norm=False,
            device=DEVICE,
        )

    def forward(self, nodes_and_graph_embedding):

        graph_embedding, nodes_embedding = torch.split(
            nodes_and_graph_embedding, [1, nodes_and_graph_embedding.shape[0] - 1]
        )

        value = self.critic(graph_embedding)

        possible_s_a_pairs = self.compute_possible_s_a_pairs(
            graph_embedding, nodes_embedding
        )
        probabilities = self.actor(possible_s_a_pairs)
        pi = F.softmax(probabilities, dim=1)

        # Just to test, not correct
        pi = torch.cat(
            (pi.squeeze(), torch.zeros(MAX_N_NODES * MAX_N_NODES - pi.shape[0]))
        )
        pi = pi.reshape(1, -1)

        return pi, value

    def compute_possible_s_a_pairs(self, graph_embedding, nodes_embedding):
        # We create 3 tensors representing state, node 1 and node 2
        # and then stack them together to get all state action pairs
        n_nodes = nodes_embedding.shape[0]
        states = graph_embedding.repeat(n_nodes * n_nodes, 1)
        nodes1 = nodes_embedding.repeat(n_nodes, 1)
        nodes2 = nodes_embedding.repeat_interleave(n_nodes, dim=0)
        s_a_pairs = torch.cat([states, nodes1, nodes2], dim=1)
        return s_a_pairs
