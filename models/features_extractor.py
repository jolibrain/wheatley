from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv, GATv2Conv, EGConv, PDNConv

from models.mlp import MLP
from utils.agent_observation import AgentObservation
from utils.utils import find_last_in_batch
from torch_geometric.nn.norm import GraphNorm

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
        max_n_machines,
        n_mlp_layers_features_extractor,
        n_layers_features_extractor,
        hidden_dim_features_extractor,
        n_attention_heads,
        reverse_adj,
        residual=True,
        normalize=True,
        conflicts_edges=False,
    ):

        self.residual = residual
        self.normalize = normalize
        self.max_n_nodes = max_n_nodes
        features_dim = input_dim_features_extractor * 2 + hidden_dim_features_extractor * (n_layers_features_extractor + 1)
        features_dim *= 2
        self.max_n_machines = max_n_machines
        super(FeaturesExtractor, self).__init__(
            observation_space=observation_space,
            features_dim=features_dim,
        )
        self.freeze_graph = freeze_graph
        self.device = device
        self.reverse_adj = reverse_adj

        self.hidden_dim_features_extractor = hidden_dim_features_extractor
        self.conflicts_edges = conflicts_edges
        self.gconv_type = gconv_type
        self.graph_has_relu = graph_has_relu
        self.graph_pooling = graph_pooling
        self.n_layers_features_extractor = n_layers_features_extractor
        self.features_extractors = nn.ModuleList()

        if self.normalize:
            self.norms = nn.ModuleList()
            self.normsbis = nn.ModuleList()

        self.embedder = MLP(
            n_layers=n_mlp_layers_features_extractor,
            input_dim=input_dim_features_extractor,
            hidden_dim=hidden_dim_features_extractor,
            output_dim=hidden_dim_features_extractor,
            batch_norm=False,
            activation="elu",
            device=self.device,
        )

        if self.gconv_type == "gatv2":
            self.mlps = nn.ModuleList()

        if self.normalize:
            self.norm0 = GraphNorm(input_dim_features_extractor)
            self.norm1 = GraphNorm(hidden_dim_features_extractor)

        for layer in range(self.n_layers_features_extractor):

            if self.normalize:
                self.norms.append(GraphNorm(hidden_dim_features_extractor))
                self.normsbis.append(GraphNorm(hidden_dim_features_extractor))

            if self.gconv_type == "gin":
                self.features_extractors.append(
                    GINConv(
                        MLP(
                            n_layers=n_mlp_layers_features_extractor,
                            input_dim=hidden_dim_features_extractor,
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
                        in_channels=hidden_dim_features_extractor,
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

            elif self.gconv_type == "eg":
                self.features_extractors.append(
                    EGConv(
                        in_channels=hidden_dim_features_extractor,
                        out_channels=hidden_dim_features_extractor,
                        aggregators=["sum", "mean", "symnorm", "min", "max", "var", "std"],
                    )
                )

            elif self.gconv_type == "pdn":
                self.features_extractors.append(
                    PDNConv(
                        in_channels=hidden_dim_features_extractor,
                        out_channels=hidden_dim_features_extractor,
                        edge_dim=1,
                        hidden_channels=16,
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

        n_batch = int(torch.max(graph_state.batch).item()) + 1

        if self.conflicts_edges or self.graph_pooling == "learn":
            first_in_batch = [0]
            last_in_batch = []
            for bi in range(n_batch):
                last_in_batch.append(find_last_in_batch(first_in_batch[bi], bi, graph_state.batch))
                first_in_batch.append(last_in_batch[bi] + 1)

        if self.conflicts_edges:
            # add bidirectional edges in case of machine conflict in order for GNN to be able
            # to pass messages
            ei0 = []
            ei1 = []
            for bi in range(n_batch):
                for ni1 in range(first_in_batch[bi], last_in_batch[bi] + 1):
                    aff1 = features[ni1, :1].item() == 1.0
                    machine1 = features[ni1, 8 : 8 + self.max_n_machines]
                    for ni2 in range(ni1 + 1, last_in_batch[bi] + 1):
                        aff2 = features[ni2, :1].item() == 1.0
                        machine2 = features[ni2, 8 : 8 + self.max_n_machines]
                        # one_hot machine id is mandatory, starts at 8, goes until 8 + max_n_machines
                        # add link only if not already present (neither nodes are planned)
                        if torch.equal(machine1, machine2) and (not (aff1 and aff2)):
                            ei0.extend([ni1, ni2])
                            ei1.extend([ni2, ni1])
            if ei0 and ei1:
                edge_index_0 = torch.cat([edge_index[0], torch.LongTensor(ei0).to(edge_index[0].device)])
                edge_index_1 = torch.cat([edge_index[1], torch.LongTensor(ei1).to(edge_index[0].device)])
                edge_index = torch.stack([edge_index_0, edge_index_1])

        if self.graph_pooling == "learn":
            graphnode = torch.zeros((n_batch, graph_state.x.shape[1])).to(features.device)
            ei0 = []
            ei1 = []
            for i in range(n_batch):
                ei0 += [n_nodes + i] * (last_in_batch[i] - first_in_batch[i] + 1)
                ei1 += list(range(first_in_batch[i], last_in_batch[i] + 1))

            edge_index_0 = torch.cat([edge_index[0], torch.LongTensor(ei0 + ei1).to(features.device)])
            edge_index_1 = torch.cat([edge_index[1], torch.LongTensor(ei1 + ei0).to(features.device)])
            edge_index = torch.stack([edge_index_0, edge_index_1])

            features = torch.cat([features, graphnode], dim=0)
        if not self.reverse_adj:
            edge_index = torch.stack([edge_index[1], edge_index[0]])

        # Compute graph embeddings
        features_list = [features]

        if self.normalize:
            features = self.norm0(features, graph_state.batch)
        features_list.append(features)
        features = self.embedder(features)
        if self.normalize:
            features = self.norm1(features, graph_state.batch)
        features_list.append(features)

        for layer in range(self.n_layers_features_extractor):
            features = self.features_extractors[layer](features, edge_index)
            if self.gconv_type == "gatv2":
                features = self.mlps[layer](features)
            if self.graph_has_relu:
                features = torch.nn.functional.elu(features)
            if self.normalize:
                features = self.norms[layer](features, graph_state.batch)
            features_list.append(features)
            if self.residual:
                features += features_list[-2]
                if self.normalize:
                    features = self.normsbis[layer](features, graph_state.batch)
        features = torch.cat(features_list, axis=1)  # The final embedding is concatenation of all layers embeddings

        if self.graph_pooling == "learn":
            graph_embedding = features[n_nodes * batch_size :, :]
            features = features[: n_nodes * batch_size, :]

        features = features.reshape(batch_size, n_nodes, -1)

        # Create graph embedding and concatenate
        if self.graph_pooling == "max":
            max_elts, max_ind = torch.max(features, dim=1)
            graph_embedding = max_elts
        elif self.graph_pooling == "avg":
            graph_pooling = torch.ones(n_nodes, device=self.device) / n_nodes
            graph_embedding = torch.matmul(graph_pooling, features)
        elif self.graph_pooling == "learn":
            pass
        else:
            raise Exception(f"Graph pooling {self.graph_pooling} not recognized. Only accepted pooling are max and avg")

        graph_embedding = graph_embedding.reshape(batch_size, 1, -1)

        # repeat the graph embedding to match the nodes embedding size
        repeated = graph_embedding.expand(features.shape)
        return torch.cat((features, repeated), dim=2)
