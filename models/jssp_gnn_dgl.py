#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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


import dgl
import torch
from dgl import LaplacianPE
from dgl.nn import DGNConv, EGATConv, GCN2Conv, GINEConv, PNAConv

from models.mlp import MLP
from utils.jssp_agent_observation import JSSPAgentObservation as AgentObservation


class JSSPGnnDGL(torch.nn.Module):
    def __init__(
        self,
        input_dim_features_extractor,
        gconv_type,
        graph_pooling,
        graph_has_relu,
        max_n_nodes,
        max_n_machines,
        n_mlp_layers_features_extractor,
        n_layers_features_extractor,
        hidden_dim_features_extractor,
        activation_features_extractor,
        n_attention_heads,
        residual=True,
        normalize=False,
        conflicts="att",
    ):
        super().__init__()
        self.conflicts = conflicts
        self.residual = residual
        self.normalize = normalize
        self.max_n_nodes = max_n_nodes
        self.features_dim = (
            input_dim_features_extractor
            + hidden_dim_features_extractor * (n_layers_features_extractor + 1)
        )
        self.features_dim *= 2
        self.max_n_machines = max_n_machines

        self.hidden_dim = hidden_dim_features_extractor
        self.gconv_type = gconv_type
        self.graph_has_relu = graph_has_relu
        self.graph_pooling = graph_pooling
        self.n_layers_features_extractor = n_layers_features_extractor
        self.features_extractors = torch.nn.ModuleList()

        self.edge_embedder = torch.nn.ModuleList()
        # self_loops 0
        # precedencies 1
        # rev recedenceies 2
        # graph pooling 3
        # graph pooling rev 4
        # optional conflicts_edges   from 5 to self.max_n_machines+5 + reverse in case of node conflict

        self.node_embedder = torch.nn.Embedding(
            1 + self.max_n_machines, input_dim_features_extractor
        )

        if self.conflicts == "clique":
            nmachineid = self.max_n_machines
        elif self.conflicts == "node":
            nmachineid = self.max_n_machines * 2
        if self.conflicts in ["clique", "node"]:
            # for layer in range(self.n_layers_features_extractor):
            #     if self.gconv_type in ["gcn2"]:
            #         self.edge_embedder.append(torch.nn.Embedding(5 + nmachineid + 1, 1))
            #     else:
            #         self.edge_embedder.append(torch.nn.Embedding(5 + nmachineid + 1, hidden_dim_features_extractor))
            if self.gconv_type in ["gcn2"]:
                self.edge_embedder.append(torch.nn.Embedding(5 + nmachineid + 1, 1))
            else:
                self.edge_embedder.append(
                    torch.nn.Embedding(
                        5 + nmachineid + 1, hidden_dim_features_extractor
                    )
                )

        else:
            # for layer in range(self.n_layers_features_extractor):
            #     if self.gconv_type in ["gcn2"]:
            #         self.edge_embedder.append(torch.nn.Embedding(5, 1))
            #     else:
            #         self.edge_embedder.append(torch.nn.Embedding(5, hidden_dim_features_extractor))
            if self.gconv_type in ["gcn2"]:
                self.edge_embedder.append(torch.nn.Embedding(5, 1))
            else:
                self.edge_embedder.append(
                    torch.nn.Embedding(5, hidden_dim_features_extractor)
                )

        self.embedder = MLP(
            n_layers=n_mlp_layers_features_extractor,
            input_dim=input_dim_features_extractor,
            hidden_dim=hidden_dim_features_extractor,
            output_dim=hidden_dim_features_extractor,
            batch_norm=self.normalize,
            activation=activation_features_extractor,
        )

        if self.normalize:
            self.norms = torch.nn.ModuleList()
            self.normsbis = torch.nn.ModuleList()
            self.norm0 = torch.nn.BatchNorm1d(input_dim_features_extractor)
            self.norm1 = torch.nn.BatchNorm1d(hidden_dim_features_extractor)

        if self.gconv_type in ["gatv2", "gat"]:
            self.mlps = torch.nn.ModuleList()
            # self.mlps_edges = torch.nn.ModuleList()

        for layer in range(self.n_layers_features_extractor):
            if self.normalize:
                self.norms.append(torch.nn.BatchNorm1d(self.hidden_dim))
                if self.residual:
                    self.normsbis.append(torch.nn.BatchNorm1d(self.hidden_dim))

            if self.gconv_type == "gin":
                mlp = MLP(
                    n_layers=n_mlp_layers_features_extractor,
                    input_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    batch_norm=self.normalize,
                    activation=activation_features_extractor,
                )
                self.features_extractors.append(GINEConv(mlp, learn_eps=True))
            elif self.gconv_type in ["gat", "gatv2"]:
                self.features_extractors.append(
                    EGATConv(
                        self.hidden_dim,
                        self.hidden_dim,
                        self.hidden_dim,
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                    )
                )
                self.mlps.append(
                    MLP(
                        n_layers=n_mlp_layers_features_extractor,
                        input_dim=hidden_dim_features_extractor * n_attention_heads,
                        hidden_dim=hidden_dim_features_extractor * n_attention_heads,
                        output_dim=hidden_dim_features_extractor,
                        batch_norm=self.normalize,
                        activation=activation_features_extractor,
                    )
                )
                # self.mlps_edges.append(
                #     MLP(
                #         n_layers=n_mlp_layers_features_extractor,
                #         input_dim=hidden_dim_features_extractor * n_attention_heads,
                #         hidden_dim=hidden_dim_features_extractor * n_attention_heads,
                #         output_dim=hidden_dim_features_extractor,
                #         batch_norm=self.normalize,
                #         activation=activation_features_extractor,
                #     )
                # )
            elif self.gconv_type == "pna":
                self.features_extractors.append(
                    PNAConv(
                        self.hidden_dim,
                        self.hidden_dim,
                        [
                            "mean",
                            "max",
                            "min",
                            "std",
                            "var",
                            "sum",
                            "moment3",
                            "moment4",
                            "moment5",
                        ],
                        ["identity", "amplification", "attenuation"],
                        2.5,
                    )
                )
            elif self.gconv_type == "dgn":
                self.features_extractors.append(
                    DGNConv(
                        self.hidden_dim,
                        self.hidden_dim,
                        [
                            "mean",
                            "max",
                            "min",
                            "std",
                            "var",
                            "sum",
                            "moment3",
                            "moment4",
                            "moment5",
                            "dir1-av",
                            "dir1-dx",
                        ],
                        ["identity", "amplification", "attenuation"],
                        2.5,
                        num_towers=int(self.hidden_dim / 4),
                    )
                )
                self.transform = LaplacianPE(k=2, feat_name="eig")
            elif self.gconv_type == "gcn2":
                self.features_extractors.append(GCN2Conv(self.hidden_dim, layer + 1))

            else:
                print("Unknown gconv type ", self.gconv_type)
                sys.exit()

    def forward(self, obs):
        observation = AgentObservation.from_gym_observation(
            obs, conflicts=self.conflicts, max_n_machines=self.max_n_machines
        )
        batch_size = observation.get_batch_size()
        n_nodes = observation.get_n_nodes()

        g = observation.to_graph()

        features = g.ndata["feat"]

        num_nodes = g.num_nodes()
        node_offset = num_nodes

        origbnn = g.batch_num_nodes()
        if self.graph_pooling == "learn":
            poolnodes = list(range(num_nodes, num_nodes + batch_size))
            g.add_nodes(
                batch_size, data={"feat": torch.zeros((batch_size, features.shape[1]))}
            )
            ei0 = []
            ei1 = []
            startnode = 0
            for i in range(batch_size):
                ei0 += [num_nodes + i] * origbnn[i]
                ei1 += list(range(startnode, startnode + origbnn[i]))
                startnode += origbnn[i]
            # REMOVE INVERSE POOLING !
            # g.add_edges(ei0, ei1, data={"type": torch.LongTensor([3] * len(ei0)).to(features.device)})

            g.add_edges(
                list(range(num_nodes, num_nodes + batch_size)),
                list(range(num_nodes, num_nodes + batch_size)),
                data={"type": torch.LongTensor([0] * batch_size)},
            )

            g.add_edges(ei1, ei0, data={"type": torch.LongTensor([4] * len(ei0))})
            node_offset += batch_size

        features = g.ndata["feat"]

        if self.conflicts == "node":
            machineid = features[:num_nodes, 6].long()
            batchid = []
            for i, nn in enumerate(origbnn):
                batchid.extend([i] * nn)
            batchid = torch.IntTensor(batchid)
            g.add_nodes(
                self.max_n_machines * batch_size,
                data={
                    "feat": torch.zeros(
                        (batch_size * self.max_n_machines, features.shape[1])
                    )
                },
            )
            machinenodeindex = list(
                range(node_offset, node_offset + self.max_n_machines * batch_size)
            )
            idxaffected = torch.where(machineid != -1, 1, 0).nonzero(as_tuple=True)[0]
            machineid = machineid[idxaffected]
            bid = batchid[idxaffected]
            targetmachinenode = bid * self.max_n_machines + machineid + node_offset
            g.add_edges(
                machinenodeindex,
                machinenodeindex,
                data={"type": torch.LongTensor([0] * self.max_n_machines * batch_size)},
            )
            g.add_edges(idxaffected, targetmachinenode, data={"type": machineid + 5})
            g.add_edges(
                targetmachinenode,
                idxaffected,
                data={"type": machineid + 5 + self.max_n_machines},
            )
            if self.graph_pooling == "learn":
                # also pool machine nodes
                machinepoolindex = []
                for i in range(batch_size):
                    machinepoolindex.extend([num_nodes + i] * self.max_n_machines)
                g.add_edges(
                    machinenodeindex,
                    machinepoolindex,
                    data={
                        "type": torch.LongTensor(
                            [5 + 2 * self.max_n_machines] * len(machinepoolindex)
                        )
                    },
                )

        g = g.to(next(self.parameters()).device)
        features = g.ndata["feat"]

        if self.graph_pooling == "learn":
            features[poolnodes] = self.node_embedder(
                torch.LongTensor([0] * len(poolnodes)).to(features.device)
            )

        if self.conflicts == "node":
            features[machinenodeindex] = self.node_embedder(
                (
                    torch.LongTensor(list(range(self.max_n_machines)) * batch_size) + 1
                ).to(features.device)
            )
        features_list = []
        if self.normalize:
            features = self.norm0(features)
        features_list.append(features)
        features = self.embedder(features)
        if self.normalize:
            features = self.norm1(features)
        features_list.append(features)

        if self.gconv_type == "dgn":
            g = self.transform(g)

        # update edge feautes below
        g.edata["emb"] = self.edge_embedder[0](g.edata["type"])

        for layer in range(self.n_layers_features_extractor):
            # do not update edge features below
            # g.edata["emb"] = self.edge_embedder[layer](g.edata["type"])
            if self.gconv_type == "gcn2":
                g.edata["emb"] = (
                    g.edata["emb"] + torch.max(torch.abs(g.edata["emb"])) + 1.0
                )
            if self.gconv_type in ["gat", "gatv2"]:
                features, new_e_attr = self.features_extractors[layer](
                    g, features, g.edata["emb"]
                )
                features = self.mlps[layer](features.flatten(start_dim=-2, end_dim=-1))
                # update edge features below
                # g.edata["emb"] = self.mlps_edges[layer](new_e_attr.flatten(start_dim=-2, end_dim=-1))
            elif self.gconv_type in ["gin", "pna"]:
                features = self.features_extractors[layer](g, features, g.edata["emb"])
            elif self.gconv_type == "dgn":
                features = self.features_extractors[layer](
                    g, features, edge_feat=g.edata["emb"], eig_vec=g.ndata["eig"]
                )
            elif self.gconv_type == "gcn2":
                features = self.features_extractors[layer](
                    g, features, features_list[1], edge_weight=g.edata["emb"].squeeze(1)
                )
            else:
                print("Unknown gconv type ", self.gconv_type)
                sys.exit()
            if self.normalize:
                features = self.norms[layer](features)
            if self.residual:
                features += features_list[-1]
                if self.normalize:
                    features = self.normsbis[layer](features)
            features_list.append(features)

        features = torch.cat(
            features_list, axis=1
        )  # The final embedding is concatenation of all layers embeddings
        node_features = features[:num_nodes, :]
        node_features = node_features.reshape(batch_size, n_nodes, -1)

        if self.graph_pooling == "max":
            max_elts, max_ind = torch.max(node_features, dim=1)
            graph_embedding = max_elts
        elif self.graph_pooling == "avg":
            graph_pooling = torch.ones(n_nodes, device=features.device) / n_nodes
            graph_embedding = torch.matmul(graph_pooling, node_features)
        elif self.graph_pooling == "learn":
            graph_embedding = features[num_nodes : num_nodes + batch_size, :]
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
