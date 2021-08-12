import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.conv import GINConv

from models.mlp import MLP
from utils.state import State

from config import (
    MAX_N_NODES,
    HIDDEN_DIM_FEATURE_EXTRACTOR,
    N_LAYERS_FEATURE_EXTRACTOR,
    N_MLP_LAYERS_FEATURE_EXTRACTOR,
    INPUT_DIM_FEATURE_EXTRACTOR,
    DEVICE,
)


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(FeaturesExtractor, self).__init__(
            observation_space=observation_space, features_dim=1
        )  # Only so it's not 0. Cf stable_baselines3 implementation

        self.n_layers_feature_extractor = N_LAYERS_FEATURE_EXTRACTOR
        self.feature_extractors = nn.ModuleList()

        for layer in range(self.n_layers_feature_extractor - 1):
            self.feature_extractors.append(
                GINConv(
                    MLP(
                        n_layers=N_MLP_LAYERS_FEATURE_EXTRACTOR,
                        input_dim=INPUT_DIM_FEATURE_EXTRACTOR
                        if layer == 0
                        else HIDDEN_DIM_FEATURE_EXTRACTOR,
                        hidden_dim=HIDDEN_DIM_FEATURE_EXTRACTOR,
                        output_dim=HIDDEN_DIM_FEATURE_EXTRACTOR,
                        batch_norm=True,
                        device=DEVICE,
                    )
                )
            )

    def forward(self, observation):
        """
        Returns the embedding of the graph concatenated with the embeddings of the nodes
        Note : the output may depend on the number of nodes, but it should not be a
        problem.
        """
        state = State.from_observation(observation)
        batch_size = state.get_batch_size()
        n_nodes = state.get_n_nodes()

        graph_state = state.to_graph()
        features, edge_index = graph_state.x, graph_state.edge_index

        for layer in range(self.n_layers_feature_extractor - 1):
            features = self.feature_extractors[layer](features, edge_index)
        features = features.reshape(batch_size, n_nodes, -1)

        # Create graph embedding and concatenate
        graph_pooling = torch.ones(n_nodes, device=DEVICE) / n_nodes
        graph_embedding = torch.matmul(graph_pooling, features)
        graph_and_nodes_embedding = torch.cat(
            (graph_embedding.reshape(batch_size, 1, -1), features), dim=1
        )

        return graph_and_nodes_embedding
