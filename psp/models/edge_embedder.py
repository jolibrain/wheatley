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
import torch
from generic.mlp import MLP


class PspEdgeEmbedder(torch.nn.Module):
    def __init__(
        self,
        edge_embedding_flavor,
        max_n_resources,
        hidden_dim,
        add_rp_edges,
        factored_rp,
        n_mlp_layers_features_extractor,
        activation_features_extractor,
    ):
        super().__init__()
        self.edge_embedding_flavor = edge_embedding_flavor
        self.max_n_resources = max_n_resources
        self.hidden_dim = hidden_dim
        self.add_rp_edges = add_rp_edges
        self.factored_rp = factored_rp

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
        # resrouce node edges 10, 11
        # we also have attributes
        # for resource conflicts : rid, rval (normalized)
        # for resource prioiries : rid, level, critical, timetype
        # type , rid, rval, timetype
        # for prec edges : only type
        # for rc edges : type,  rid, rval
        # for rp edges : type, rid , level, criticial, timetype

        self.n_edge_type = 16

        if self.edge_embedding_flavor == "sum":
            # self.resource_id_embedder = torch.nn.Embedding(
            #     self.max_n_resources + 1, self.hidden_dim
            # )
            self.edge_type_embedder = torch.nn.Embedding(
                self.n_edge_type, self.hidden_dim
            )

            self.rc_att_embedder = torch.nn.Linear(2, self.hidden_dim)
            if self.add_rp_edges != "none":
                if self.factored_rp:
                    self.rp_att_embedder = torch.nn.Linear(
                        3 * self.max_n_resources, self.hidden_dim
                    )
                else:
                    self.rp_att_embedder = torch.nn.Linear(3, self.hidden_dim)

        elif self.edge_embedding_flavor == "cat":
            self.edge_type_embedder = torch.nn.Embedding(self.n_edge_type, 4)
            # self.resource_id_embedder = torch.nn.Embedding(
            #     self.max_n_resources + 1, self.max_n_resources + 1
            # )
            # rest = self.hidden_dim - self.n_edge_type - self.max_n_resources - 1
            rest = self.hidden_dim - 4
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
        # elif self.edge_embedding_flavor == "cartesian":
        #     self.type_rid_hidden_dim = int(self.hidden_dim / 2)
        #     self.type_rid_embedder = torch.nn.Embedding(
        #         8 * (self.max_n_resources + 1), self.type_rid_hidden_dim
        #     )
        #     self.rc_att_hidden_dim = int(
        #         (self.hidden_dim - self.type_rid_hidden_dim) / 2
        #     )
        #     self.rc_att_embedder = torch.nn.Linear(2, self.rc_att_hidden_dim)
        #     if self.add_rp_edges != "none":
        #         self.rp_att_hidden_dim = (
        #             self.hidden_dim - self.type_rid_hidden_dim - self.rc_att_hidden_dim
        #         )

        #         if self.factored_rp:
        #             self.rp_att_embedder = torch.nn.Linear(
        #                 3 * max_n_resources, self.rp_att_hidden_dim
        #             )
        #         else:
        #             self.rp_att_embedder = torch.nn.Linear(3, self.rp_att_hidden_dim)
        else:
            raise ValueError(
                "unknown edge embedding flavor " + self.edge_embedding_flavor
            )

    def forward(self, g):
        if self.edge_embedding_flavor == "sum":
            ret = self.edge_type_embedder(g.edata(None, "type"))
            # ret += self.esource_id_embedder(g.edata(None, "rid"))
            try:
                ret += self.rc_att_embedder(g.edata(None, "att_rc"))
            except KeyError:
                pass
            if self.add_rp_edges != "none":
                try:  # if no ressource priory info in graph (ie at start state), key is absent
                    ret += self.rp_att_embedder(g.edata(None, "att_rp").float())
                except KeyError:
                    pass
            return ret

        if self.edge_embedding_flavor == "cat":
            et = self.edge_type_embedder(g.edata(None, "type"))
            # ei = self.resource_id_embedder(g.edata(None, "rid"))
            try:
                ec = self.rc_att_embedder(g.edata(None, "att_rc"))
            except KeyError:
                ec = torch.zeros((g.num_edges(), self.rc_att_hidden_dim))
            if self.add_rp_edges != "none":
                try:
                    ep = self.rp_att_embedder(g.edata(None, "att_rp"))
                except KeyError:
                    ep = torch.zeros((g.num_edges(), self.rp_att_hidden_dim))
                # return torch.cat([et, ei, ec, ep], dim=-1)
                return torch.cat([et, ec, ep], dim=-1)
            # return torch.cat([et, ei, ec], dim=-1)
            return torch.cat([et, ei, ec], dim=-1)

        if self.edge_embedding_flavor == "cartesian":
            eit = self.type_rid_embedder(
                g.edata(None, "type") * (self.max_n_resources + 1)
                + g.edata(None, "rid")
            )
            try:
                ec = self.rc_att_embedder(g.edata(None, "att_rc"))
            except KeyError:
                ec = torch.zeros((g.num_edges(), self.rc_att_hidden_dim))
            if self.add_rp_edges != "none":
                try:
                    ep = self.rp_att_embedder(g.edata(None, "att_rp"))
                except KeyError:
                    ep = torch.zeros((g.num_edges(), self.rp_att_hidden_dim))
                return torch.cat([eit, ec, ep], dim=-1)
            return torch.cat([eit, ec], dim=-1)
