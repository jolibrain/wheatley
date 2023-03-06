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

import torch
from models.tokengt.utils import get_laplacian_pe_simple
from .utils import compute_conflicts_cliques, put_back_one_hot_encoding_unbatched
import dgl
import time


class AgentObservation:
    def __init__(self, graphs, glist=False):
        self.graphs = graphs
        self.glist = glist

    def get_batch_size(self):
        if self.glist:
            return len(self.graphs)
        return self.graphs.batch_size

    def get_n_nodes(self):
        if self.glist:
            return int(sum([g.num_nodes() for g in self.graphs]) / len(self.graphs))
        else:
            return int(self.graphs.num_nodes() / self.graphs.batch_size)

    @classmethod
    def build_graph(cls, n_edges, edges, nnodes, feats, bidir):
        edges0 = edges[0]
        edges1 = edges[1]
        type0 = [1] * n_edges
        if bidir:
            type1 = [2] * n_edges

            gnew = dgl.graph(
                (torch.cat([edges0, edges1]), torch.cat([edges1, edges0])),
                num_nodes=nnodes,
            )
        else:
            gnew = dgl.graph((edges0, edges1), num_nodes=nnodes)

        gnew.ndata["feat"] = feats
        if bidir:
            type0.extend(type1)
        gnew.edata["type"] = torch.LongTensor(type0)
        return gnew

    @classmethod
    def get_machine_id(cls, machine_one_hot):
        # print("machine_one_hot", machine_one_hot)

        return torch.max(machine_one_hot, dim=1)

    @classmethod
    def add_conflicts_cliques2(cls, g, cedges, mid):
        g.add_edges(cedges[0], cedges[1], data={"type": mid[0] + 5})
        return g

    @classmethod
    def add_conflicts_cliques(cls, g, features, nnodes, max_n_machines):
        machineid = features[:, 6].long()
        m1 = machineid.unsqueeze(0).expand(nnodes, nnodes)
        # put m2 unaffected to -2 so that unaffected task are not considered in conflict
        m2 = torch.where(machineid == -1, -2, machineid).unsqueeze(1).expand(nnodes, nnodes)
        cond = torch.logical_and(
            torch.eq(m1, m2),
            torch.logical_not(torch.diag(torch.BoolTensor([True] * nnodes))),
        )
        conflicts = torch.where(cond, 1, 0).nonzero(as_tuple=True)
        edgetype = machineid[conflicts[0]] + 5
        g.add_edges(conflicts[0], conflicts[1], data={"type": edgetype})
        return g

    @classmethod
    def from_gym_observation(
        cls,
        gym_observation,
        conflicts="att",
        max_n_machines=-1,
        add_self_loops=True,
        device=None,
        do_batch=True,
        compute_laplacian_pe=False,
        laplacian_pe_cache=None,
        n_laplacian_eigv=50,
        bidir=True,
    ):

        # batching on CPU for performance reasons...
        n_nodes = gym_observation["n_nodes"].long().to(torch.device("cpu"))
        n_edges = gym_observation["n_edges"].long().to(torch.device("cpu"))
        edge_index = gym_observation["edge_index"].long().to(torch.device("cpu"))
        orig_feat = gym_observation["features"]  # .to(torch.device("cpu"))

        if conflicts != "clique":
            # orig_feat = put_back_one_hot_encoding_unbatched(orig_feat, max_n_machines)
            orig_feat = orig_feat.to(torch.device("cpu"))
        else:
            if "n_conflict_edges" in gym_observation:  # precomputed cliques
                n_conflict_edges = gym_observation["n_conflict_edges"].long().to(torch.device("cpu"))

                conflicts_edges = gym_observation["conflicts_edges"].long().to(torch.device("cpu"))
                conflicts_edges_machineid = gym_observation["conflicts_edges_machineid"].long().to(torch.device("cpu"))
                orig_feat = orig_feat.to(torch.device("cpu"))
            else:  # compute cliques
                conflicts_edges = []
                conflicts_edges_machineid = []
                all_nce = []
                for i in range(orig_feat.shape[0]):
                    ce, cemid = compute_conflicts_cliques(orig_feat[i, : n_nodes[i], 6].long().squeeze(0))
                    cemid = cemid.unsqueeze_(0).expand(ce.shape)
                    conflicts_edges.append(ce.long().to(torch.device("cpu")))
                    conflicts_edges_machineid.append(cemid.long().to(torch.device("cpu")))
                    nce = torch.LongTensor([ce.shape[1]])
                    all_nce.append(nce)

                # conflicts_edges = torch.stack(all_ce).to(torch.device("cpu"))

                # Pas besoin de stack, garder la liste pour en dessous
                # conflicts_edges_machineid = (
                #     torch.stack(all_cemid).unsqueeze_(1).expand(conflicts_edges.shape).to(torch.device("cpu"))
                # )

                n_conflict_edges = torch.cat(all_nce)
                # orig_feat = put_back_one_hot_encoding_unbatched(orig_feat, max_n_machines)
                orig_feat = orig_feat.to(torch.device("cpu"))

        graphs = []
        if do_batch:
            batch_num_nodes = []
            batch_num_edges = []

        for i, nnodes in enumerate(n_nodes):
            features = orig_feat[i, :nnodes, :]
            gnew = cls.build_graph(
                n_edges[i],
                edge_index[i, :, : n_edges[i].item()],
                nnodes.item(),
                orig_feat[i, : nnodes.item(), :],
                bidir,
            )

            if conflicts == "clique":
                gnew = AgentObservation.add_conflicts_cliques2(
                    gnew,
                    conflicts_edges[i][:, : n_conflict_edges[i].item()],
                    conflicts_edges_machineid[i][:, : n_conflict_edges[i].item()],
                )
                # gnew = AgentObservation.add_conflicts_cliques(gnew, features, nnodes.item(), max_n_machines)

            if add_self_loops:
                gnew = dgl.add_self_loop(gnew, edge_feat_names=["type"], fill_data=0)
            if compute_laplacian_pe:
                gnew.ndata["laplacian_pe"] = get_laplacian_pe_simple(gnew, laplacian_pe_cache, n_laplacian_eigv)
            gnew = gnew.to(device)
            graphs.append(gnew)
            if do_batch:
                batch_num_nodes.append(gnew.num_nodes())
                batch_num_edges.append(gnew.num_edges())

        if do_batch:
            graph = dgl.batch(graphs)
            graph.set_batch_num_nodes(torch.tensor(batch_num_nodes))
            graph.set_batch_num_edges(torch.tensor(batch_num_edges))

            return cls(graph, glist=False)
        else:
            return cls(graphs, glist=True)

    def to_graph(self):
        """
        Returns the batched graph associated with the observation.
        """
        return self.graphs
