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


from models.mlp import MLP
import torch
import dgl
from dgl.nn import EGATConv

from dgl import LaplacianPE
from utils.psp_agent_observation import PSPAgentObservation as AgentObservation


class PSPGnnDGL(torch.nn.Module):
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
    ):

        super().__init__()
        self.conflicts = conflicts
        self.residual = residual
        self.normalize = normalize
        self.max_n_nodes = max_n_nodes
        self.input_dim_features_extractor = input_dim_features_extractor
        self.layer_pooling = layer_pooling
        if layer_pooling == "all":
            self.features_dim = (
                input_dim_features_extractor
                + hidden_dim_features_extractor * (n_layers_features_extractor + 1)
            )
        else:
            self.features_dim = hidden_dim_features_extractor

        self.features_dim *= 2
        self.max_n_resources = max_n_resources

        self.hidden_dim = hidden_dim_features_extractor
        self.graph_pooling = graph_pooling
        self.n_layers_features_extractor = n_layers_features_extractor
        self.features_extractors = torch.nn.ModuleList()

        # EDGES
        # types:
        # self_loops 0
        # precedencies 1
        # rev recedenceies 2
        # static ressource conflicts 3
        # resource priority 4
        # reverse resource priority 5
        # graph_pooling 6
        # reverse graph pooling 7
        # we also have attributes
        # for resource conflicts : rid, rval (normalized)
        # for resource prioiries : rid, level, critical, timetype
        # type , rid, rval, timetype
        # for prec edges : only type
        # for rc edges : type,  rid, rval
        # for rp edges : type, rid , level, criticial, timetype

        self.edge_embedding_flavor = edge_embedding_flavor
        if self.edge_embedding_flavor == "sum":
            self.resource_id_embedder = torch.nn.Embedding(
                self.max_n_resources + 1, hidden_dim_features_extractor
            )
            self.edge_type_embedder = torch.nn.Embedding(
                7, hidden_dim_features_extractor
            )

            self.rc_att_embedder = torch.nn.Linear(2, hidden_dim_features_extractor)
            self.rp_att_embedder = torch.nn.Linear(3, hidden_dim_features_extractor)
        elif self.edge_embedding_flavor == "cat":
            self.edge_type_embedder = torch.nn.Embedding(7, 7)
            self.resource_id_embedder = torch.nn.Embedding(
                self.max_n_resources + 1, self.max_n_resources + 1
            )
            rest = hidden_dim_features_extractor - 7 - self.max_n_resources - 1
            if rest < 4:
                raise ValueError(
                    f"too small hidden_dim_features_extractor for cat edge embedder, should be at least max_n_resources + num_edge_type + 4, ie {self.max_n_resources+11}"
                )
            self.rc_att_hidden_dim = int(rest / 2)
            self.rp_att_hidden_dim = rest - self.rc_att_hidden_dim
            self.rc_att_embedder = torch.nn.Linear(2, self.rc_att_hidden_dim)
            self.rp_att_embedder = torch.nn.Linear(3, self.rp_att_hidden_dim)
        elif self.edge_embedding_flavor == "cartesian":
            self.type_rid_hidden_dim = int(hidden_dim_features_extractor / 2)
            self.type_rid_embedder = torch.nn.Embedding(
                7 * (self.max_n_resources + 1), self.type_rid_hidden_dim
            )
            self.rc_att_hidden_dim = int(
                (hidden_dim_features_extractor - self.type_rid_hidden_dim) / 2
            )
            self.rp_att_hidden_dim = (
                hidden_dim_features_extractor
                - self.type_rid_hidden_dim
                - self.rc_att_hidden_dim
            )
            self.rc_att_embedder = torch.nn.Linear(2, self.rc_att_hidden_dim)
            self.rp_att_embedder = torch.nn.Linear(3, self.rp_att_hidden_dim)
        else:
            raise ValueError(
                "unknown edge embedding flavor " + self.edge_embedding_flavor
            )

        self.pool_node_embedder = torch.nn.Embedding(1, input_dim_features_extractor)

        self.features_embedder = MLP(
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

        self.mlps = torch.nn.ModuleList()

        for layer in range(self.n_layers_features_extractor):

            if self.normalize:
                self.norms.append(torch.nn.BatchNorm1d(self.hidden_dim))
                if self.residual:
                    self.normsbis.append(torch.nn.BatchNorm1d(self.hidden_dim))

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

    def embed_edges(self, g):
        if self.edge_embedding_flavor == "sum":
            ret = self.edge_type_embedder(g.edata["type"])
            ret += self.resource_id_embedder(g.edata["rid"])
            try:
                ret += self.rc_att_embedder(g.edata["att_rc"])
            except KeyError:
                pass
            try:  # if no ressource priory info in graph (ie at start state), key is absent
                ret += self.rp_att_embedder(g.edata["att_rp"].float())
            except KeyError:
                pass
            return ret

        if self.edge_embedding_flavor == "cat":
            et = self.edge_type_embedder(g.edata["type"])
            ei = self.resource_id_embedder(g.edata["rid"])
            try:
                ec = self.rc_att_embedder(g.edata["att_rc"])
            except KeyError:
                ec = torch.zeros((g.num_edges(), self.rc_att_hidden_dim))
            try:
                ep = self.rp_att_embedder(g.edata["att_rp"])
            except KeyError:
                ep = torch.zeros((g.num_edges(), self.rp_att_hidden_dim))
            return torch.cat([et, ei, ec, ep], dim=-1)

        if self.edge_embedding_flavor == "cartesian":
            eit = self.type_rid_embedder(
                g.edata["type"] * (self.max_n_resources + 1) + g.edata["rid"]
            )
            try:
                ec = self.rc_att_embedder(g.edata["att_rc"])
            except KeyError:
                ec = torch.zeros((g.num_edges(), self.rc_att_hidden_dim))
            try:
                ep = self.rp_att_embedder(g.edata["att_rp"])
            except KeyError:
                ep = torch.zeros((g.num_edges(), self.rp_att_hidden_dim))
            return torch.cat([eit, ec, ep], dim=-1)

    def forward(self, obs):

        observation = AgentObservation.from_gym_observation(
            obs, conflicts=self.conflicts, add_self_loops=False
        )
        batch_size = observation.get_batch_size()
        n_nodes = observation.get_n_nodes()

        g = observation.to_graph()

        num_nodes = g.num_nodes()
        node_offset = num_nodes

        origbnn = g.batch_num_nodes()
        if self.graph_pooling == "learn":
            poolnodes = list(range(num_nodes, num_nodes + batch_size))
            g.add_nodes(
                batch_size,
                data={
                    "feat": torch.zeros((batch_size, self.input_dim_features_extractor))
                },
            )
            ei0 = []
            ei1 = []
            startnode = 0
            for i in range(batch_size):
                ei0 += [num_nodes + i] * origbnn[i]
                ei1 += list(range(startnode, startnode + origbnn[i]))
                startnode += origbnn[i]

            g.add_edges(
                list(range(num_nodes, num_nodes + batch_size)),
                list(range(num_nodes, num_nodes + batch_size)),
                data={"type": torch.LongTensor([0] * batch_size)},
            )

            g.add_edges(ei1, ei0, data={"type": torch.LongTensor([6] * len(ei0))})
            # INVERSE POOLING BELOW !
            # g.add_edges(ei0, ei1, data={"type": torch.LongTensor([7] * len(ei0))})
            node_offset += batch_size

        g = g.to(next(self.parameters()).device)
        features = g.ndata["feat"]

        if self.graph_pooling == "learn":
            features[poolnodes] = self.pool_node_embedder(
                torch.LongTensor([0] * len(poolnodes)).to(features.device)
            )

        if self.layer_pooling == "all":
            features_list = []
        if self.normalize:
            features = self.norm0(features)
        if self.layer_pooling == "all":
            features_list.append(features)
        features = self.features_embedder(features)
        if self.normalize:
            features = self.norm1(features)
        if self.layer_pooling == "all":
            features_list.append(features)

        # update edge feautes below
        g.edata["emb"] = self.embed_edges(g)

        if self.layer_pooling == "last":
            previous_feat = features

        for layer in range(self.n_layers_features_extractor):
            # do not update edge features below
            # g.edata["emb"] = self.edge_embedder[layer](g.edata["type"])
            features, new_e_attr = self.features_extractors[layer](
                g, features, g.edata["emb"]
            )
            features = self.mlps[layer](features.flatten(start_dim=-2, end_dim=-1))
            # update edge features below
            # g.edata["emb"] = self.mlps_edges[layer](new_e_attr.flatten(start_dim=-2, end_dim=-1))
            if self.normalize:
                features = self.norms[layer](features)
            if self.residual:
                if self.layer_pooling == "all":
                    features += features_list[-1]
                else:
                    features += previous_feat
                if self.normalize:
                    features = self.normsbis[layer](features)
                if self.layer_pooling == "last":
                    previous_feat = features
            if self.layer_pooling == "all":
                features_list.append(features)

        if self.layer_pooling == "all":
            features = torch.cat(
                features_list, axis=1
            )  # The final embedding is concatenation of all layers embeddings
        node_features = features[:num_nodes, :]
        node_features = node_features.reshape(batch_size, n_nodes, -1)

        if self.graph_pooling == "max":
            max_elts, max_ind = torch.max(node_features, dim=1)
            graph_embedding = max_elts
        elif self.graph_pooling == "avg":
            graph_pooling = torch.ones(n_nodes, device=self.device) / n_nodes
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
