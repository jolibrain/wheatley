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
    ):
        super().__init__()
        self.residual = residual
        self.normalize = normalize
        self.max_n_nodes = max_n_nodes
        self.layer_pooling = layer_pooling
        self.hierarchical = hierarchical
        self.shared_layers = shared_layers

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
            features[nodesid[ntype], :] = nembedder(g, nodesid[ntype].to(device))
            embeded[nodesid[ntype]] = True
        assert torch.all(embeded)

        edge_features = torch.empty((num_edges, self.hidden_dim), device=device)
        edge_embedded = torch.zeros(num_edges, dtype=bool)
        for strtype, edge_embedder in self.edge_embedders.items():
            etype = strtype_to_eetype(strtype)
            if etype in edgesid and edgesid[etype].numel() != 0:
                edge_features[edgesid[etype], :] = edge_embedder(
                    g, edgesid[etype].to(device)
                )
                edge_embedded[edgesid[etype]] = True

        assert torch.all(edge_embedded)

        #        orig_features = features

        features, poolnodes_features = self.gnn(
            g, features, edge_features, nodesid["n"]
        )

        # if self.layer_pooling == "all":
        #     features = torch.cat([orig_features] + features, dim=-1)
        #     if poolnodes_features is not None:
        #         poolnodes_features = torch.cat(poolnodes_features, dim=-1)

        node_features = features[nodesid["n"], :]

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
            node_features = node_features.reshape(batch_size, n_nodes[0], -1)

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

            node_features = torch.nn.functional.pad(
                node_features,
                (0, 0, 0, self.max_n_nodes - node_features.shape[1]),
                mode="constant",
                value=0.0,
            )

        graph_embedding = graph_embedding.reshape(batch_size, -1)

        return node_features, graph_embedding
