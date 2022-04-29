from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv, GATv2Conv

from models.mlp import MLP
from utils.agent_observation import AgentObservation

import sys


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        input_dim_features_extractor,
        gconv_type,
        graph_pooling,
        freeze_graph,
        graph_has_relu,
        device,
        max_n_nodes,
        n_mlp_layers_features_extractor,
        n_layers_features_extractor,
        hidden_dim_features_extractor,
        n_attention_heads,
    ):
        self.max_n_nodes = max_n_nodes
        super(FeaturesExtractor, self).__init__(
            observation_space=observation_space,
            features_dim=(
                (hidden_dim_features_extractor * n_layers_features_extractor + input_dim_features_extractor)
                + self.max_n_nodes
            )
            * (self.max_n_nodes + 1),
        )
        self.freeze_graph = freeze_graph
        self.device = device

        self.gconv_type = gconv_type
        self.graph_has_relu = graph_has_relu
        self.graph_pooling = graph_pooling
        self.n_layers_features_extractor = n_layers_features_extractor
        self.features_extractors = nn.ModuleList()

        if self.gconv_type == "gatv2":
            self.mlps = nn.ModuleList()

        for layer in range(self.n_layers_features_extractor):
            if self.gconv_type == "gin":
                self.features_extractors.append(
                    GINConv(
                        MLP(
                            n_layers=n_mlp_layers_features_extractor,
                            input_dim=input_dim_features_extractor if layer == 0 else hidden_dim_features_extractor,
                            hidden_dim=hidden_dim_features_extractor,
                            output_dim=hidden_dim_features_extractor,
                            batch_norm=False if self.freeze_graph else True,
                            activation="relu",
                            device=self.device,
                        )
                    )
                )

            elif self.gconv_type == "gatv2":
                self.features_extractors.append(
                    GATv2Conv(
                        in_channels=input_dim_features_extractor if layer == 0 else hidden_dim_features_extractor,
                        out_channels=hidden_dim_features_extractor,
                        heads=n_attention_heads,
                    )
                )
                self.mlps.append(
                    MLP(
                        n_layers=n_mlp_layers_features_extractor,
                        input_dim=hidden_dim_features_extractor * n_attention_heads,
                        hidden_dim=hidden_dim_features_extractor * n_attention_heads,
                        output_dim=hidden_dim_features_extractor,
                        batch_norm=False,
                        activation="elu",
                        device=self.device,
                    )
                )

            else:
                print("Unknown gconv type ", self.gconv_type)
                sys.exit()

            self.features_extractors[-1].to(self.device)

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

        graph_state = observation.to_torch_geometric()
        features, edge_index = graph_state.x, graph_state.edge_index

        # Compute graph embeddings
        features_list = [features]
        for layer in range(self.n_layers_features_extractor):
            features = self.features_extractors[layer](features, edge_index)
            if self.gconv_type == "gatv2":
                features = self.mlps[layer](features)
            if self.graph_has_relu:
                features = torch.nn.functional.elu(features)
            features_list.append(features)
        features = torch.cat(features_list, axis=1)  # The final embedding is concatenation of all layers embeddings
        features = features.reshape(batch_size, n_nodes, -1)

        # Create graph embedding and concatenate
        if self.graph_pooling == "max":
            max_elts, max_ind = torch.max(features, dim=1)
            graph_embedding = max_elts
        elif self.graph_pooling == "avg":
            graph_pooling = torch.ones(n_nodes, device=self.device) / n_nodes
            graph_embedding = torch.matmul(graph_pooling, features)
        else:
            raise Exception(f"Graph pooling {self.graph_pooling} not recognized. Only accepted pooling are max and avg")
        graph_and_nodes_embedding = torch.cat((graph_embedding.reshape(batch_size, 1, -1), features), dim=1)

        return graph_and_nodes_embedding
