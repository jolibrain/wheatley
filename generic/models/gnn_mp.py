#
# Wheatley
# Copyright (c) 2026 Jolibrain
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

from .gnn_flat import GnnFlat
from .gnn_hier import GnnHier
# from .gnn_tgp import GnnTGP


def eetype_to_strtype(eetype):
    return eetype[0] + "&>" + eetype[1] + "&>" + eetype[2]


def strtype_to_eetype(strtype):
    return tuple(strtype.split("&>"))


class GnnMP(torch.nn.Module):
    def __init__(
        self,
        node_embedders_spec,
        edge_embedders_spec,
        graph_pooling,
        max_n_nodes,
        n_mlp_layers_features_extractor,
        n_layers_features_extractor,
        hidden_dim_features_extractor,
        activation_features_extractor,
        n_attention_heads,
        num_node_types=2,
        residual=True,
        normalize=False,
        layer_pooling="all",
        vnode=False,
        hierarchical=False,
        checkpoint=1,
        do_compile=True,
        gconv_activation="swiglu",
        shared_layers=False,
        g2=False,
        decision_nodes="n",
    ):
        super().__init__()
        self.residual = residual
        self.normalize = normalize
        self.max_n_nodes = max_n_nodes
        self.layer_pooling = layer_pooling
        self.hierarchical = hierarchical
        self.shared_layers = shared_layers
        self.decision_nodes = decision_nodes

        self.node_type_size = num_node_types
        if do_compile:
            torch._dynamo.config.capture_scalar_outputs = True

        if layer_pooling == "all" and self.hierarchical:
            # self.features_dim = hidden_dim_features_extractor * (
            #     n_layers_features_extractor * 2 + 3
            # )
            self.features_dim = hidden_dim_features_extractor
        else:
            self.features_dim = hidden_dim_features_extractor

        self.hidden_dim = hidden_dim_features_extractor
        self.graph_pooling = graph_pooling
        self.n_layers_features_extractor = n_layers_features_extractor

        self.checkpoint = checkpoint

        self.node_embedders = torch.nn.ModuleDict()
        for netype, nedata in node_embedders_spec.items():
            ne = nedata["class"](
                hidden_dim_features_extractor,
                **nedata["kwargs"],
            )
            if do_compile:
                ne = torch.compile(ne, dynamic=True)
            self.node_embedders[netype] = ne
        self.edge_embedders = torch.nn.ModuleDict()
        for eetype, eedata in edge_embedders_spec.items():
            ee = eedata["class"](
                hidden_dim_features_extractor,
                **eedata["kwargs"],
            )
            if do_compile:
                ee = torch.compile(ee, dynamic=True)
            self.edge_embedders[eetype_to_strtype(eetype)] = ee

        if self.hierarchical:
            self.gnn = GnnHier(
                # self.gnn = GnnTGP(
                hidden_dim=hidden_dim_features_extractor,
                n_layers=n_layers_features_extractor,
                n_mlp_layers=n_mlp_layers_features_extractor,
                layer_pooling=layer_pooling,
                n_attention_heads=n_attention_heads,
                normalize=normalize,
                activation=activation_features_extractor,
                residual=residual,
                checkpoint=self.checkpoint,
                graph_pooling=graph_pooling,
                gconv_activation=gconv_activation,
                shared_layers=self.shared_layers,
                g2=g2,
            )

        else:
            self.gnn = GnnFlat(
                hidden_dim=hidden_dim_features_extractor,
                n_layers=n_layers_features_extractor,
                n_mlp_layers=n_mlp_layers_features_extractor,
                layer_pooling=layer_pooling,
                n_attention_heads=n_attention_heads,
                normalize=normalize,
                activation=activation_features_extractor,
                residual=residual,
                checkpoint=self.checkpoint,
                gconv_activation=gconv_activation,
                shared_layers=self.shared_layers,
                g2=g2,
            )
            if do_compile:
                self.gnn = torch.compile(self.gnn, dynamic=True)

    def reset_egat(self):
        for egat in self.features_extractors:
            egat.reset_parameters()

    def forward(self, obs):
        (
            g,
            native_aos,
            batch_size,
            num_nodes,
            num_edges,
            n_nodes,
            n_edges,
            nodesid,
            edgesid,
            nodes_types_map,
            edges_types_map,
        ) = obs

        device = next(self.parameters()).device
        g = g.to(device)

        features = torch.empty((num_nodes, self.hidden_dim), device=device)
        embeded = torch.zeros(num_nodes, dtype=bool)

        for ntype, nembedder in self.node_embedders.items():
            if ntype in nodesid:
                features[nodesid[ntype], :] = nembedder(g, nodesid[ntype].to(device))
                embeded[nodesid[ntype]] = True
        if not torch.all(embeded):
            not_embedded = torch.where(torch.logical_not(embeded))[0]
            types_not_embedded = g.node_type[not_embedded]
            uniques_types = torch.unique(types_not_embedded)
            for i in uniques_types:
                typeid = i.item()
                for nt in nodes_types_map:
                    if nodes_types_map[nt] == typeid:
                        print(f"node type not embedded : {nt}")
            exit(1)

        edge_features = torch.empty((num_edges, self.hidden_dim), device=device)
        edge_embedded = torch.zeros(num_edges, dtype=bool)
        for strtype, edge_embedder in self.edge_embedders.items():
            etype = strtype_to_eetype(strtype)
            if etype in edgesid and edgesid[etype].numel() != 0:
                edge_features[edgesid[etype], :] = edge_embedder(
                    g, edgesid[etype].to(device)
                )
                edge_embedded[edgesid[etype]] = True

        if not torch.all(edge_embedded):
            not_embedded = torch.where(torch.logical_not(edge_embedded))[0]
            types_not_embedded = g.edge_type[not_embedded]
            uniques_types = torch.unique(types_not_embedded)
            for i in uniques_types:
                typeid = i.item()
                for et in edges_types_map:
                    if edges_types_map[et] == typeid:
                        print(f"edge type not embedded : {et}")
            exit(1)

        #        orig_features = features

        features, poolnodes_features = self.gnn(
            g, features, edge_features, nodesid[self.decision_nodes]
        )

        # if self.layer_pooling == "all":
        #     features = torch.cat([orig_features] + features, dim=-1)
        #     if poolnodes_features is not None:
        #         poolnodes_features = torch.cat(poolnodes_features, dim=-1)

        node_features = features[nodesid[self.decision_nodes], :]

        if batch_size != 1:
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
                if poolnodes_features is None:
                    graph_embedding = features[nodesid["poolnode"], :]
                else:
                    graph_embedding = poolnodes_features

        else:
            node_features = self.unbatch(node_features, native_aos)

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
                if poolnodes_features is None:
                    graph_embedding = features[nodesid["poolnode"], :]
                else:
                    graph_embedding = poolnodes_features
            else:
                raise Exception(
                    f"Graph pooling {self.graph_pooling} not recognized. Only accepted pooling are max and avg"
                )

        graph_embedding = graph_embedding.reshape(batch_size, -1)

        return node_features, graph_embedding

    def unbatch(self, nf, aos):
        start = 0
        nnf = []
        for ao in aos:
            ndn = ao.g.num_nodes(self.decision_nodes)
            nnf.append(nf[start : start + ndn])
            start += ndn
        return nnf
