from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv, GATv2Conv

from models.mlp import MLP
from utils.agent_observation import AgentObservation

from config import (
    MAX_N_NODES,
    HIDDEN_DIM_FEATURES_EXTRACTOR,
    N_LAYERS_FEATURES_EXTRACTOR,
    N_MLP_LAYERS_FEATURES_EXTRACTOR,
    DEVICE,
)

import sys


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, input_dim_features_extractor, gconv_type, max_pool, freeze_graph, graph_has_relu):
        super(FeaturesExtractor, self).__init__(
            observation_space=observation_space,
            features_dim=(HIDDEN_DIM_FEATURES_EXTRACTOR + MAX_N_NODES) * (MAX_N_NODES + 1),
        )
        self.freeze_graph = freeze_graph

        self.gconv_type = gconv_type
        self.graph_has_relu = graph_has_relu
        self.max_pool = max_pool
        self.n_layers_features_extractor = N_LAYERS_FEATURES_EXTRACTOR
        self.features_extractors = nn.ModuleList()

        for layer in range(self.n_layers_features_extractor):
            if self.gconv_type == "gin":
                self.features_extractors.append(
                    GINConv(
                        MLP(
                            n_layers=N_MLP_LAYERS_FEATURES_EXTRACTOR,
                            input_dim=input_dim_features_extractor if layer == 0 else HIDDEN_DIM_FEATURES_EXTRACTOR,
                            hidden_dim=HIDDEN_DIM_FEATURES_EXTRACTOR,
                            output_dim=HIDDEN_DIM_FEATURES_EXTRACTOR,
                            batch_norm=False if self.freeze_graph else True,
                            activation="relu",
                            device=DEVICE,
                        )
                    )
                )

            elif self.gconv_type == "gatv2":
                self.features_extractors.append(
                    GATv2Conv(
                        in_channels=input_dim_features_extractor if layer == 0 else HIDDEN_DIM_FEATURES_EXTRACTOR,
                        out_channels=HIDDEN_DIM_FEATURES_EXTRACTOR,
                    )
                )

            else:
                print("Unknown gconv type ", self.gconv_type)
                sys.exit()

            self.features_extractors[-1].to(DEVICE)

        if self.freeze_graph:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, obs):
        """
        Returns the embedding of the graph concatenated with the embeddings of the nodes
        Note : the output may depend on the number of nodes, but it should not be a
        problem.
        """
        observation = AgentObservation.from_gym_observation(obs)
        batch_size = observation.get_batch_size()
        n_nodes = observation.get_n_nodes()
        mask = observation.get_mask()

        graph_state = observation.to_torch_geometric()
        features, edge_index = graph_state.x, graph_state.edge_index

        # Compute graph embeddings
        for layer in range(self.n_layers_features_extractor):
            features = self.features_extractors[layer](features, edge_index)
            if self.graph_has_relu:
                features = torch.nn.functional.elu(features)
        features = features.reshape(batch_size, n_nodes, -1)

        # Create graph embedding and concatenate
        if self.max_pool:
            max_elts, max_ind = torch.max(features,dim=1)
            graph_embedding = max_elts
        else: # avg
            graph_pooling = torch.ones(n_nodes, device=DEVICE) / n_nodes
            graph_embedding = torch.matmul(graph_pooling, features)
        graph_and_nodes_embedding = torch.cat((graph_embedding.reshape(batch_size, 1, -1), features), dim=1)

        mask = mask.reshape(batch_size, n_nodes, n_nodes)
        extended_mask = torch.cat((torch.zeros((batch_size, 1, n_nodes), device=DEVICE), mask), dim=1)
        embedded_features = torch.cat((graph_and_nodes_embedding, extended_mask), dim=2)

        return embedded_features
