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
from dgl.nn import GlobalAttentionPooling

# from generic.eegatconv import EEGATConv
from psp.graph.graph_conv import GraphConv

from dgl import LaplacianPE
from .agent_observation import AgentObservation
from .agent_graph_observation import AgentGraphObservation
from .edge_embedder import PspEdgeEmbedder
from .rewiring import rewire, homogeneous_edges
from .gnn_flat import GnnFlat
from .gnn_hier import GnnHier


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
        checkpoint=1,
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
        self.features_dim *= 2
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

        if self.rwpe_k != 0:
            self.rwpe_embedder = torch.nn.Linear(self.rwpe_k * 4, self.rwpe_h)

        self.pool_node_embedder = torch.nn.Embedding(
            1, self.input_dim_features_extractor
        )

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
                1, self.input_dim_features_extractor
            )
        if self.conflicts == "node":
            # self.resource_node_embedder = torch.nn.Embedding(
            #     max_n_resources, self.input_dim_features_extractor
            # )
            self.resource_node_embedder = torch.nn.Linear(
                3, self.input_dim_features_extractor
            )

        self.edge_embedder = PspEdgeEmbedder(
            edge_embedding_flavor,
            self.max_n_resources,
            self.hidden_dim,
            self.add_rp_edges,
            self.factored_rp,
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
                pyg,
                self.checkpoint,
            )

    def reset_egat(self):
        for egat in self.features_extractors:
            egat.reset_parameters()

    def forward(self, obs):
        if self.graphobs:
            observation = AgentGraphObservation(
                obs,
                conflicts=self.conflicts,
                factored_rp=self.factored_rp,
                add_rp_edges=self.add_rp_edges,
                max_n_resources=self.max_n_resources,
                rwpe_k=self.rwpe_k,
                rwpe_cache=self.rwpe_cache,
                rewire_internal=False,
            )
            g = observation.graphs
            res_cal_id = observation.res_cal_id
            batch_size = observation.n_graphs
            num_nodes = observation.num_nodes
            if batch_size == 1:
                n_nodes = observation.batch_num_nodes[0]
            else:
                n_nodes = observation.batch_num_nodes
            origbnn = observation.batch_num_nodes
        else:
            observation = AgentObservation(
                obs,
                conflicts=self.conflicts,
                add_self_loops=False,
                factored_rp=self.factored_rp,
                add_rp_edges=self.add_rp_edges,
                max_n_resources=self.max_n_resources,
                rwpe_k=self.rwpe_k,
                rwpe_cache=self.rwpe_cache,
            )
            batch_size = observation.get_batch_size()

            g = observation.to_graph()
            res_cal_id = observation.res_cal_id.flatten()

            n_nodes = observation.get_n_nodes()
            num_nodes = g.num_nodes()

            origbnn = g.batch_num_nodes()
        if self.add_self_loops:
            if self.graphobs:
                g.add_self_loops()
            # else:
            #     g = dgl.add_self_loop(
            #         g,
            #         edge_feat_names=["type"],
            #         fill_data=AgentObservation.edgeType["self"],
            #     )
        # g.set_batch_num_nodes(origbnn)

        g, poolnodes, resource_nodes, vnodes = rewire(
            g,
            self.graph_pooling,
            self.conflicts == "node",
            self.vnode,
            batch_size,
            self.input_dim_features_extractor,
            self.max_n_resources,
            6,
            7,
            10,
            10,
            8,
            9,
            self.graphobs,
        )

        if self.graphobs:
            g, felist = homogeneous_edges(
                g,
                AgentGraphObservation.edgeTypes,
                self.factored_rp,
                self.conflicts,
                self.max_n_resources,
            )
            g.to_homogeneous(ndata=["feat"], edata=felist)

        if self.rwpe_k != 0:
            edge_features_pe = self.edge_embedder_pe(g)
            g.ndata["pe"] = self.rwpe_embedder(
                torch.cat(
                    [
                        g.ndata["rwpe_global"],
                        g.ndata["rwpe_pr"],
                        g.ndata["rwpe_rp"],
                        g.ndata["rwpe_rc"],
                    ],
                    1,
                ),
            )
            pe = g.ndata["pe"]
        else:
            pe = None

        g = g.to(next(self.parameters()).device)
        edge_features = self.edge_embedder(g)
        if self.update_edge_features:
            edge_scores = self.edge_embedder_scores(g)
        features = g.ndata("feat")
        # remove job_id and node_type
        features[:, 2:4] = 0
        # remove resource cons
        features[:, 10:] = 0

        if self.graph_pooling == "learn":
            features[poolnodes] = self.pool_node_embedder(
                torch.LongTensor([0] * len(poolnodes)).to(features.device)
            )

        if self.vnode:
            features[vnodes] = self.vnode_embedder(
                torch.LongTensor([0] * len(vnodes)).to(features.device)
            )
        if self.conflicts == "node":
            features[resource_nodes] = self.resource_node_embedder(
                # one embedding per resnode : not better, nres dependent : DISCARD
                # torch.LongTensor(list(range(self.max_n_resources)) * batch_size).to(
                # same embedding
                # torch.LongTensor([0] * batch_size * self.max_n_resources).to(
                #     features.device
                # )
                # embedding depending on calendar type
                res_cal_id.to(features.device)
            )

        features = self.gnn(g, features, edge_features, pe)

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
        repeated = graph_embedding.expand(node_features.shape)
        ret = torch.cat((node_features, repeated), dim=2)
        return ret
