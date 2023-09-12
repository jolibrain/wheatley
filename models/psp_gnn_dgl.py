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
        factored_rp=False,
        add_rp_edges="all",
        add_self_loops=False,
        vnode=False,
        update_edge_features=False,
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

        self.factored_rp = factored_rp
        self.add_rp_edges = add_rp_edges
        self.features_dim *= 2
        self.max_n_resources = max_n_resources
        self.add_self_loops = add_self_loops
        self.update_edge_features = update_edge_features

        self.hidden_dim = hidden_dim_features_extractor
        self.graph_pooling = graph_pooling
        self.n_layers_features_extractor = n_layers_features_extractor
        self.features_extractors = torch.nn.ModuleList()
        self.vnode = vnode

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
        # to/from vnode 8 , 9
        # resource node self-loop : 10 .. 10 + rnres
        # resrouce node edges 10+nres .. 10+2*nres
        # we also have attributes
        # for resource conflicts : rid, rval (normalized)
        # for resource prioiries : rid, level, critical, timetype
        # type , rid, rval, timetype
        # for prec edges : only type
        # for rc edges : type,  rid, rval
        # for rp edges : type, rid , level, criticial, timetype

        n_edge_type = 11 + self.max_n_resources * 3

        self.edge_embedding_flavor = edge_embedding_flavor
        if self.edge_embedding_flavor == "sum":
            self.resource_id_embedder = torch.nn.Embedding(
                self.max_n_resources + 1, hidden_dim_features_extractor
            )
            self.edge_type_embedder = torch.nn.Embedding(
                n_edge_type, hidden_dim_features_extractor
            )

            self.rc_att_embedder = torch.nn.Linear(2, hidden_dim_features_extractor)
            if self.add_rp_edges != "none":
                if self.factored_rp:
                    self.rp_att_embedder = torch.nn.Linear(
                        3 * self.max_n_resources, hidden_dim_features_extractor
                    )
                else:
                    self.rp_att_embedder = torch.nn.Linear(
                        3, hidden_dim_features_extractor
                    )
        elif self.edge_embedding_flavor == "cat":
            self.edge_type_embedder = torch.nn.Embedding(n_edge_type, n_edge_type)
            self.resource_id_embedder = torch.nn.Embedding(
                self.max_n_resources + 1, self.max_n_resources + 1
            )
            rest = hidden_dim_features_extractor - 12 - self.max_n_resources - 1
            if rest < 4:
                raise ValueError(
                    f"too small hidden_dim_features_extractor for cat edge embedder, should be at least max_n_resources + num_edge_type + 4, ie {self.max_n_resources+11}"
                )
            self.rc_att_hidden_dim = int(rest / 2)
            self.rc_att_embedder = torch.nn.Linear(2, self.rc_att_hidden_dim)
            if self.add_rp_edges != "none":
                self.rp_att_hidden_dim = rest - self.rc_att_hidden_dim
                if self.factored_rp:
                    self.rp_att_embedder = torch.nn.Linear(
                        3 * self.max_n_resources, self.rp_att_hidden_dim
                    )
                else:
                    self.rp_att_embedder = torch.nn.Linear(3, self.rp_att_hidden_dim)
        elif self.edge_embedding_flavor == "cartesian":
            self.type_rid_hidden_dim = int(hidden_dim_features_extractor / 2)
            self.type_rid_embedder = torch.nn.Embedding(
                8 * (self.max_n_resources + 1), self.type_rid_hidden_dim
            )
            self.rc_att_hidden_dim = int(
                (hidden_dim_features_extractor - self.type_rid_hidden_dim) / 2
            )
            self.rc_att_embedder = torch.nn.Linear(2, self.rc_att_hidden_dim)
            if self.add_rp_edges != "none":
                self.rp_att_hidden_dim = (
                    hidden_dim_features_extractor
                    - self.type_rid_hidden_dim
                    - self.rc_att_hidden_dim
                )

                if self.factored_rp:
                    self.rp_att_embedder = torch.nn.Linear(
                        3 * max_n_resources, self.rp_att_hidden_dim
                    )
                else:
                    self.rp_att_embedder = torch.nn.Linear(3, self.rp_att_hidden_dim)
        else:
            raise ValueError(
                "unknown edge embedding flavor " + self.edge_embedding_flavor
            )

        self.pool_node_embedder = torch.nn.Embedding(1, input_dim_features_extractor)
        self.vnode_embedder = torch.nn.Embedding(1, input_dim_features_extractor)
        self.resource_node_embedder = torch.nn.Embedding(
            max_n_resources, input_dim_features_extractor
        )

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
        if self.update_edge_features:
            self.mlps_edges = torch.nn.ModuleList()

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
            if self.update_edge_features:
                self.mlps_edges.append(
                    MLP(
                        n_layers=n_mlp_layers_features_extractor,
                        input_dim=hidden_dim_features_extractor * n_attention_heads,
                        hidden_dim=hidden_dim_features_extractor * n_attention_heads,
                        output_dim=hidden_dim_features_extractor,
                        batch_norm=self.normalize,
                        activation=activation_features_extractor,
                    )
                )

    def reset_egat(self):
        for egat in self.features_extractors:
            egat.reset_parameters()

    def embed_edges(
        self,
        g,
        layer,
    ):
        if self.edge_embedding_flavor == "sum":
            ret = self.edge_type_embedder(g.edata["type"])
            ret += self.resource_id_embedder(g.edata["rid"])
            try:
                ret += self.rc_att_embedder(g.edata["att_rc"])
            except KeyError:
                pass
            if self.add_rp_edges != "none":
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
            if self.add_rp_edges != "none":
                try:
                    ep = self.rp_att_embedder(g.edata["att_rp"])
                except KeyError:
                    ep = torch.zeros((g.num_edges(), self.rp_att_hidden_dim))
                return torch.cat([et, ei, ec, ep], dim=-1)
            return torch.cat([et, ei, ec], dim=-1)

        if self.edge_embedding_flavor == "cartesian":
            eit = self.type_rid_embedder(
                g.edata["type"] * (self.max_n_resources + 1) + g.edata["rid"]
            )
            try:
                ec = self.rc_att_embedder(g.edata["att_rc"])
            except KeyError:
                ec = torch.zeros((g.num_edges(), self.rc_att_hidden_dim))
            if self.add_rp_edges != "none":
                try:
                    ep = self.rp_att_embedder(g.edata["att_rp"])
                except KeyError:
                    ep = torch.zeros((g.num_edges(), self.rp_att_hidden_dim))
                return torch.cat([eit, ec, ep], dim=-1)
            return torch.cat([eit, ec], dim=-1)

    def forward(self, obs):
        observation = AgentObservation.from_gym_observation(
            obs,
            conflicts=self.conflicts,
            add_self_loops=self.add_self_loops,
            factored_rp=self.factored_rp,
            add_rp_edges=self.add_rp_edges,
            max_n_resources=self.max_n_resources,
        )
        batch_size = observation.get_batch_size()
        n_nodes = observation.get_n_nodes()

        g = observation.to_graph()

        num_nodes = g.num_nodes()
        node_offset = num_nodes

        origbnn = g.batch_num_nodes()

        if self.graph_pooling == "learn":
            poolnodes = list(range(node_offset, node_offset + batch_size))
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
                ei0 += [node_offset + i] * origbnn[i]
                ei1 += list(range(startnode, startnode + origbnn[i]))
                startnode += origbnn[i]

            g.add_edges(
                list(range(node_offset, node_offset + batch_size)),
                list(range(node_offset, node_offset + batch_size)),
                data={"type": torch.LongTensor([0] * batch_size)},
            )

            g.add_edges(ei1, ei0, data={"type": torch.LongTensor([6] * len(ei0))})
            # INVERSE POOLING BELOW !
            # g.add_edges(ei0, ei1, data={"type": torch.LongTensor([7] * len(ei0))})
            node_offset += batch_size

        if self.conflicts == "node":
            resources_used = g.ndata["feat"][:, 10:]
            num_resources = resources_used.shape[1]
            resource_nodes = list(
                range(node_offset, node_offset + num_resources * batch_size)
            )
            batch_id = []
            for i, nn in enumerate(origbnn):
                batch_id.extend([i] * nn)
            batch_id = torch.IntTensor(batch_id)
            g.add_nodes(
                num_resources * batch_size,
                data={
                    "feat": torch.zeros(
                        (
                            batch_size * self.max_n_resources,
                            self.input_dim_features_extractor,
                        )
                    )
                },
            )
            idxaffected = torch.where(resources_used != 0)
            consumers = idxaffected[0]
            nconsumers = consumers.shape[0]
            resource_start_per_batch = []
            for i in range(batch_size):
                resource_start_per_batch.append(node_offset + num_resources * i)
            resource_start_per_batch = torch.IntTensor(resource_start_per_batch)
            resource_index = (
                idxaffected[1] + resource_start_per_batch[batch_id[consumers]]
            )

            rntype = torch.LongTensor(
                list(range(num_resources)) * batch_size
            ) + torch.LongTensor([10] * len(resource_nodes))

            g.add_edges(
                resource_nodes,
                resource_nodes,
                # we could use different self loop type per resource
                data={"type": rntype},
                # data={"type": torch.LongTensor([10] * len(resource_nodes))},
            )
            rc = torch.gather(
                resources_used[consumers], 1, idxaffected[1].unsqueeze(1)
            ).expand(nconsumers, 2)

            g.add_edges(
                consumers,
                resource_index,
                data={
                    # "type": torch.LongTensor([10] * nconsumers),
                    "type": torch.LongTensor([10 + num_resources] * nconsumers)
                    + idxaffected[1].long(),
                    "rid": idxaffected[1].int(),
                    "att_rc": rc,
                },
            )
            g.add_edges(
                resource_index,
                consumers,
                data={
                    # "type": torch.LongTensor([11] * nconsumers),
                    "type": torch.LongTensor([10 + 2 * num_resources] * nconsumers)
                    + idxaffected[1].long(),
                    "rid": idxaffected[1].int(),
                    "att_rc": rc,
                },
            )
            node_offset += num_resources * batch_size

        if self.vnode:
            vnodes = list(range(node_offset, node_offset + batch_size))
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
                ei0 += [node_offset + i] * origbnn[i]
                ei1 += list(range(startnode, startnode + origbnn[i]))
                startnode += origbnn[i]

            # g.add_edges(
            #     list(range(node_offset, node_offset + batch_size)),
            #     list(range(node_offset, node_offset + batch_size)),
            #     data={"type": torch.LongTensor([0] * batch_size)},
            # )

            g.add_edges(ei1, ei0, data={"type": torch.LongTensor([8] * len(ei0))})
            g.add_edges(ei0, ei1, data={"type": torch.LongTensor([9] * len(ei0))})
            node_offset += batch_size

        g = g.to(next(self.parameters()).device)
        features = g.ndata["feat"]

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
                torch.LongTensor(list(range(num_resources)) * batch_size).to(
                    features.device
                )
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
        edge_features = self.embed_edges(g, 0)

        if self.layer_pooling == "last":
            previous_feat = features

        torch.autograd.set_detect_anomaly(True)
        for layer in range(self.n_layers_features_extractor):
            # do not update edge features below
            if self.update_edge_features:
                features, new_e_attr = self.features_extractors[layer](
                    g, features, edge_features.clone()
                )
            else:
                features, new_e_attr = self.features_extractors[layer](
                    g, features, edge_features
                )

            features = self.mlps[layer](features.flatten(start_dim=-2, end_dim=-1))
            # update edge features below
            if self.update_edge_features:
                edge_features += self.mlps_edges[layer](
                    new_e_attr.flatten(start_dim=-2, end_dim=-1)
                )
            # if self.layer_pooling == "all":
            #     features_list.append(features)
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
