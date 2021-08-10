import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.conv import GINConv

from models.mlp import MLP

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
        Return the embedding of the graph concatenated with the embeddings of the nodes.
        Note : the output may depend on the number of nodes, but it should not be a
        problem.
        """
        # First determine if observation is batched or single
        batched = True if len(list(observation["n_nodes"].shape)) > 2 else False

        # Convert from 1 hot encoding to actual number
        if batched:
            # Note : we consider n_nodes to be constant across batch
            n_nodes = (observation["n_nodes"][0][0] == 1).nonzero(as_tuple=True)[0]
        else:
            n_nodes = (observation["n_nodes"][0] == 1).nonzero(as_tuple=True)[0]
        n_nodes = n_nodes.item()

        batch_size = features.shape[0]
        features = observation["features"]
        edge_index = observation["edge_index"].long()
        features = features[:, 0:n_nodes]

        # Extract features
        data_list = [
            Data(x=features[i], edge_index=edge_index[i]) for i in range(batch_size)
        ]
        loader = DataLoader(data_list, batch_size=batch_size)
        aggregated_data = loader.next()
        aggregated_features = aggregated_data.x
        aggregated_edge_index = aggregated_data.edge_index

        for layer in range(self.n_layers_feature_extractor - 1):
            aggregated_features = self.feature_extractors[layer](
                aggregated_features, aggregated_edge_index
            )
        features = aggreagted_features.reshape(batch_size, n_nodes, -1)

        # TODO : adapt this to batches
        # Create graph embedding and concatenate
        graph_pooling = torch.ones(n_nodes) / n_nodes
        graph_embedding = torch.matmul(graph_pooling, features)
        graph_and_nodes_embedding = torch.cat(
            (graph_embedding.reshape(1, -1), features)
        )

        return graph_and_nodes_embedding
