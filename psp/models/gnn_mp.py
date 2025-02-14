#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#


from generic.mlp import MLP
import torch
# from dgl.nn import GlobalAttentionPooling

from psp.graph.graph_conv import GraphConv
from .agent_graph_observation import AgentGraphObservation
from .edge_embedder import PspEdgeEmbedder
from .rewiring import rewire, homogeneous_edges
from .gnn_flat import GnnFlat
from .gnn_hier import GnnHier
from generic.tokenGT import TokenGT


class GnnMP(torch.nn.Module):
    def __init__(
        self,
        input_dim_features_extractor,
        graph_pooling,
        max_n_nodes,
        max_n_resources,
        n_mlp_layers_features_extractor,
        n_layers_features_extractor,
        hidden_dim_features_extractor,
        activation_features_extractor,
        n_attention_heads,
        residual=True,
        normalize=False,
        conflicts="att",
        edge_embedding_flavor="sum",
        layer_pooling="all",
        factored_rp=False,
        add_rp_edges="all",
        add_self_loops=False,
        vnode=False,
        update_edge_features=False,
        update_edge_features_pe=False,
        rwpe_k=16,
        rwpe_h=16,
        rwpe_cache=None,
        graphobs=False,
        shared_conv=True,
        pyg=True,
        hierarchical=False,
        tokengt=False,
        checkpoint=1,
        dual_net=False,
    ):
        super().__init__()
        self.pyg = pyg
        self.conflicts = conflicts
        self.residual = residual
        self.normalize = normalize
        self.max_n_nodes = max_n_nodes
        self.input_dim_features_extractor = input_dim_features_extractor
        self.layer_pooling = layer_pooling
        self.rwpe_k = rwpe_k
        if self.rwpe_k == 0:
            self.rwpe_h = 0
        else:
            self.rwpe_h = rwpe_h
        self.hierarchical = hierarchical

        self.node_type_size = 2
        self.tokengt = tokengt
        if self.tokengt:
            layer_pooling = "last"

        if layer_pooling == "all":
            self.features_dim = (
                self.input_dim_features_extractor
                + (hidden_dim_features_extractor + self.rwpe_h)
                * (n_layers_features_extractor + 1)
                + self.rwpe_h
            )
        else:
            self.features_dim = hidden_dim_features_extractor + self.rwpe_h

        self.graphobs = graphobs

        self.factored_rp = factored_rp
        self.add_rp_edges = add_rp_edges
        self.max_n_resources = max_n_resources
        # self.add_self_loops = add_self_loops
        self.add_self_loops = True
        self.update_edge_features = update_edge_features
        self.update_edge_features_pe = update_edge_features_pe

        self.hidden_dim = hidden_dim_features_extractor
        self.graph_pooling = graph_pooling
        self.n_layers_features_extractor = n_layers_features_extractor

        self.vnode = vnode
        self.shared_conv = shared_conv
        self.checkpoint = checkpoint

        self.rwpe_cache = rwpe_cache
        self.edge_embedding_flavor = edge_embedding_flavor
        self.dual_net = dual_net

        if self.rwpe_k != 0:
            self.rwpe_embedder = torch.nn.Linear(self.rwpe_k * 4, self.rwpe_h)

        # 0 for poolnode
        # 1 for ressource
        # 2 for task
        # 3 for vnode
        self.node_type_embedder = torch.nn.Embedding(4, self.node_type_size)
        self.node_type_embedder = torch.compile(self.node_type_embedder)

        self.pool_node_embedder = torch.nn.Embedding(
            1,
            hidden_dim_features_extractor - self.node_type_size,
        )
        self.pool_node_embedder = torch.compile(self.pool_node_embedder)

        if self.dual_net:
            self.node_type_embedder2 = torch.nn.Embedding(4, self.node_type_size)
            self.node_type_embedder2 = torch.compile(self.node_type_embedder2)

            self.pool_node_embedder2 = torch.nn.Embedding(
                1,
                hidden_dim_features_extractor - self.node_type_size,
            )
            self.pool_node_embedder2 = torch.compile(self.pool_node_embedder2)

        if self.graph_pooling == "gap":
            self.gate_nn = MLP(
                n_layers=1,
                input_dim=self.features_dim // 2,
                hidden_dim=self.features_dim // 2,
                output_dim=1,
                norm=self.normalize,
                activation=activation_features_extractor,
            )
            self.gap = GlobalAttentionPooling(self.gate_nn)
        if self.vnode:
            self.vnode_embedder = torch.nn.Embedding(
                1,
                hidden_dim_features_extractor - self.node_type_size,
            )
            if self.dual_net:
                self.vnode_embedder2 = torch.nn.Embedding(
                    1,
                    hidden_dim_features_extractor - self.node_type_size,
                )

        if self.conflicts == "node":
            self.resource_node_embedder = torch.nn.Linear(
                3,
                hidden_dim_features_extractor - self.node_type_size,
            )
            self.resource_node_embedder = torch.compile(self.resource_node_embedder)
            if self.dual_net:
                self.resource_node_embedder2 = torch.nn.Linear(
                    3,
                    hidden_dim_features_extractor - self.node_type_size,
                )
                self.resource_node_embedder2 = torch.compile(
                    self.resource_node_embedder2
                )

        if self.update_edge_features:
            self.edge_embedder_scores = PspEdgeEmbedder(
                edge_embedding_flavor,
                self.max_n_resources,
                self.hidden_dim,
                self.add_rp_edges,
                self.factored_rp,
            )

        if self.rwpe_k != 0:
            self.edge_embedder_pe = PspEdgeEmbedder(
                edge_embedding_flavor,
                self.max_n_resources,
                self.hidden_dim,
                self.add_rp_edges,
                self.factored_rp,
            )

        if self.hierarchical:
            self.gnn = GnnHier(
                self.input_dim_features_extractor,
                hidden_dim_features_extractor,
                n_layers_features_extractor,
                n_mlp_layers_features_extractor,
                layer_pooling,
                n_attention_heads,
                normalize,
                activation_features_extractor,
                True,
                rwpe_k,
                self.rwpe_h,
                update_edge_features,
                update_edge_features_pe,
                shared_conv,
                pyg,
                self.checkpoint,
            )
        elif self.tokengt:
            self.gnn = TokenGT(
                self.input_dim_features_extractor,
                hidden_dim_features_extractor,
                hidden_dim_features_extractor,
                n_layers_features_extractor,
                n_attention_heads,
                "orf",
                10,
                10,
                use_graph_token=False,
            )

        else:
            self.gnn = GnnFlat(
                self.input_dim_features_extractor,
                hidden_dim_features_extractor,
                n_layers_features_extractor,
                n_mlp_layers_features_extractor,
                layer_pooling,
                n_attention_heads,
                normalize,
                activation_features_extractor,
                residual,
                rwpe_k,
                self.rwpe_h,
                update_edge_features,
                update_edge_features_pe,
                shared_conv,
                edge_embedding_flavor,
                max_n_resources,
                add_rp_edges,
                factored_rp,
                pyg,
                self.checkpoint,
            )
            self.gnn = torch.compile(self.gnn, dynamic=True)
            if self.dual_net:
                self.gnn2 = GnnFlat(
                    self.input_dim_features_extractor,
                    hidden_dim_features_extractor,
                    n_layers_features_extractor,
                    n_mlp_layers_features_extractor,
                    layer_pooling,
                    n_attention_heads,
                    normalize,
                    activation_features_extractor,
                    residual,
                    rwpe_k,
                    self.rwpe_h,
                    update_edge_features,
                    update_edge_features_pe,
                    shared_conv,
                    edge_embedding_flavor,
                    max_n_resources,
                    add_rp_edges,
                    factored_rp,
                    pyg,
                    self.checkpoint,
                )
                self.gnn2 = torch.compile(self.gnn2, dynamic=True)

        self.features_embedder = MLP(
            n_layers=n_mlp_layers_features_extractor,
            input_dim=self.input_dim_features_extractor,
            hidden_dim=self.hidden_dim - self.node_type_size,
            output_dim=self.hidden_dim - self.node_type_size,
            norm=self.normalize,
            activation=activation_features_extractor,
        )
        self.features_embedder = torch.compile(self.features_embedder, dynamic=True)
        if self.dual_net:
            self.features_embedder2 = MLP(
                n_layers=n_mlp_layers_features_extractor,
                input_dim=self.input_dim_features_extractor,
                hidden_dim=self.hidden_dim - self.node_type_size,
                output_dim=self.hidden_dim - self.node_type_size,
                norm=self.normalize,
                activation=activation_features_extractor,
            )
            self.features_embedder2 = torch.compile(
                self.features_embedder2, dynamic=True
            )

    def reset_egat(self):
        for egat in self.features_extractors:
            egat.reset_parameters()

    def forward(self, obs):
        (
            g,
            batch_size,
            num_nodes,
            n_nodes,
            poolnodes,
            resource_nodes,
            vnodes,
            res_cal_id,
            pe,
        ) = obs

        g = g.to(next(self.parameters()).device)

        if self.update_edge_features:
            edge_scores = self.edge_embedder_scores(g)
        features = g.ndata("feat")
        # remove job_id and node_type
        features[:, 2:4] = 0

        if self.normalize:
            features = self.norm0(features)
        orig_features = features

        features = torch.empty(
            (orig_features.shape[0], self.hidden_dim), device=orig_features.device
        )

        features[:num_nodes, : self.node_type_size] = self.node_type_embedder(
            torch.LongTensor([2] * num_nodes).to(features.device)
        )
        features[:num_nodes, self.node_type_size :] = self.features_embedder(
            orig_features[:num_nodes]
        )

        if self.graph_pooling == "learn":
            features[poolnodes, : self.node_type_size] = self.node_type_embedder(
                torch.LongTensor([0] * len(poolnodes)).to(features.device)
            )
            features[poolnodes, self.node_type_size :] = self.pool_node_embedder(
                torch.LongTensor([0] * len(poolnodes)).to(features.device)
            )

        if self.dual_net:
            features2 = torch.empty(
                (orig_features.shape[0], self.hidden_dim), device=orig_features.device
            )

            features2[:num_nodes, : self.node_type_size] = self.node_type_embedder2(
                torch.LongTensor([2] * num_nodes).to(features.device)
            )
            features2[:num_nodes, self.node_type_size :] = self.features_embedder2(
                orig_features[:num_nodes]
            )

            if self.graph_pooling == "learn":
                features2[poolnodes, : self.node_type_size] = self.node_type_embedder2(
                    torch.LongTensor([0] * len(poolnodes)).to(features.device)
                )
                features2[poolnodes, self.node_type_size :] = self.pool_node_embedder2(
                    torch.LongTensor([0] * len(poolnodes)).to(features.device)
                )

        if self.vnode:
            features[vnodes, : self.node_type_size] = self.node_type_embedder(
                torch.LongTensor([3] * len(vnodes)).to(features.device)
            )
            features[vnodes, self.node_type_size :] = self.vnode_embedder(
                torch.LongTensor([0] * len(vnodes)).to(features.device)
            )
            if self.dual_net:
                features2[vnodes, : self.node_type_size] = self.node_type_embedder2(
                    torch.LongTensor([3] * len(vnodes)).to(features.device)
                )
                features2[vnodes, self.node_type_size :] = self.vnode_embedder2(
                    torch.LongTensor([0] * len(vnodes)).to(features.device)
                )

        if self.conflicts == "node":
            features[resource_nodes, : self.node_type_size] = self.node_type_embedder(
                torch.LongTensor([1] * len(resource_nodes)).to(features.device)
            )
            features[resource_nodes, self.node_type_size :] = (
                self.resource_node_embedder(res_cal_id.to(device=features.device))
            )
            if self.dual_net:
                features2[resource_nodes, : self.node_type_size] = (
                    self.node_type_embedder2(
                        torch.LongTensor([1] * len(resource_nodes)).to(features.device)
                    )
                )
                features2[resource_nodes, self.node_type_size :] = (
                    self.resource_node_embedder2(res_cal_id.to(device=features.device))
                )

        features = self.gnn(g, features, pe)
        if self.dual_net:
            features2 = self.gnn2(g, features2, pe)

        if self.layer_pooling == "all":
            if self.rwpe_k != 0:
                features = torch.cat([orig_features, pe] + features, dim=-1)
            else:
                features = torch.cat([orig_features] + features, dim=-1)
                if self.dual_net:
                    features2 = torch.cat([orig_features] + features2, dim=-1)

        node_features = features[:num_nodes, :]

        if self.graphobs and batch_size != 1:
            if self.graph_pooling == "max":
                graph_embedding = []
                startelt = 0
                for i in range(batch_size):
                    nn = n_nodes[i]
                    graph_embedding.append(
                        torch.max(node_features[startelt : startelt + nn], dim=0)[0]
                    )
                    startelt += nn
                graph_embedding = torch.stack(graph_embedding)
            elif self.graph_pooling == "avg":
                graph_embedding = []
                startelt = 0
                for i in range(batch_size):
                    nn = n_nodes[i]
                    gp = torch.ones(nn, device=node_features.device) / nn
                    graph_embedding.append(
                        torch.matmul(gp, node_features[startelt : startelt + nn])
                    )
                    startelt += nn
                graph_embedding = torch.stack(graph_embedding)
            elif self.graph_pooling in ["learn", "learninv"]:
                if self.dual_net:
                    graph_embedding = features2[poolnodes, :]
                else:
                    graph_embedding = features[poolnodes, :]

            nnf = []
            startelt = 0
            for i in range(batch_size):
                nn = n_nodes[i]
                nnf.append(
                    torch.nn.functional.pad(
                        node_features[startelt : startelt + nn],
                        (0, 0, 0, self.max_n_nodes - nn),
                        mode="constant",
                        value=0.0,
                    )
                )
                startelt += nn
            node_features = torch.stack(nnf)

        else:
            node_features = node_features.reshape(batch_size, n_nodes, -1)

            if self.graph_pooling == "max":
                max_elts, _ = torch.max(node_features, dim=1)
                graph_embedding = max_elts
            elif self.graph_pooling == "avg":
                graph_pooling = (
                    torch.ones(n_nodes, device=node_features.device) / n_nodes
                )
                graph_embedding = torch.matmul(graph_pooling, node_features)
            elif self.graph_pooling == "gap":
                graph_embedding = self.gap(g, features)
            elif self.graph_pooling in ["learn", "learninv"]:
                if self.dual_net:
                    graph_embedding = features2[poolnodes, :]
                else:
                    graph_embedding = features[poolnodes, :]
            else:
                raise Exception(
                    f"Graph pooling {self.graph_pooling} not recognized. Only accepted pooling are max and avg"
                )

            node_features = torch.nn.functional.pad(
                node_features,
                (0, 0, 0, self.max_n_nodes - node_features.shape[1]),
                mode="constant",
                value=0.0,
            )

        graph_embedding = graph_embedding.reshape(batch_size, 1, -1)
        # repeat the graph embedding to match the nodes embedding size
        # repeated = graph_embedding.clone().detach().expand(node_features.shape)
        # ret = torch.cat((node_features, repeated), dim=2)
        # return ret
        return node_features, graph_embedding
        # return ret, graph_embedding
