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
import numpy as np
from generic.tokengt.utils import get_laplacian_pe_simple
from psp.utils.utils import compute_resources_graph_torch
import dgl
import time


class AgentGraphObservation:
    edgeTypes = {
        "self": 0,
        "prec": 1,
        "rprec": 2,
        "rc": 3,
        "rp": 4,
        "rrp": 5,
        "pool": 6,
        "rpool": 7,
        "vnode": 8,
        "rvnode": 9,
        "nodeconf": 10,
        "rnodeconf": 11,
    }

    @classmethod
    def rewire_internal(
        cls,
        g,
        conflicts="clique",
        bidir=True,
        compute_laplacian_pe=False,
        laplacian_pe_cache=None,
        rwpe_k=0,
        rwpe_cache=None,
    ):
        g.ndata["feat"] = torch.cat(
            [
                g.ndata["affected"].unsqueeze(1),
                g.ndata["selectable"].unsqueeze(1),
                g.ndata["job"].unsqueeze(1),
                g.ndata["type"].unsqueeze(1),
                g.ndata["normalized_durations"],
                g.ndata["normalized_tct"],
                g.ndata["resources"],
            ],
            dim=1,
        )
        if conflicts == "clique":
            if g.num_edges(etype="rc") == 0:
                (
                    resource_conf_edges,
                    resource_conf_id,
                    resource_conf_val,
                    resource_conf_val_r,
                ) = compute_resources_graph_torch(g.ndata["resources"])
                g.add_edges(
                    resource_conf_edges[0],
                    resource_conf_edges[1],
                    data={
                        "rid": resource_conf_id,
                        "val": resource_conf_val,
                        "valr": resource_conf_val_r,
                    },
                    etype="rc",
                )

        if bidir:
            prec_edges = g.edges(etype="prec")
            g.add_edges(prec_edges[1], prec_edges[0], etype="rprec")

            if g.num_edges(etype="rp") != 0:
                rp_edges = g.edges(etype="rp")
                g.add_edges(
                    rp_edges[1],
                    rp_edges[0],
                    etype="rrp",
                    data={"r": g.edata["r"][("n", "rp", "n")]},
                )

        # if add_self_loops:
        #     g = dgl.add_self_loop(
        #         gnew,
        #         edge_feat_names=["type"],
        #         etype="self",
        #         fill_data=AgentObservation.edgeType["self"],
        #     )
        if compute_laplacian_pe:
            g.ndata["laplacian_pe"] = get_laplacian_pe_simple(
                dgl.to_homogeneous(g, store_type=False, return_count=False),
                laplacian_pe_cache,
                n_laplacian_eigv,
            )

        if rwpe_k != 0:
            g.ndata["rwpe_global"] = cls.rwpe(
                dgl.to_homogeneous(g, store_type=False, return_count=False),
                rwpe_k,
                rwpe_cache,
            )
            g.ndata["rwpe_pr"] = self.rwpe(
                dgl.to_homogeneous(
                    g.edge_type_subgraph(["prec", "rprec"]),
                    store_type=False,
                    return_count=False,
                ),
                rwpe_k,
                rwpe_cache,
            )
            g.ndata["rwpe_rp"] = self.rwpe(
                dgl.to_homogeneous(
                    g.edge_type_subgraph(["rp", "rrp"]),
                    store_type=False,
                    return_count=False,
                ),
                rwpe_k,
                rwpe_cache,
            )
            g.ndata["rwpe_rc"] = self.rwpe(
                dgl.to_homogeneous(
                    g.edge_type_subgraph(["rc"]),
                    store_type=False,
                    return_count=False,
                ),
                rwpe_k,
                rwpe_cache,
            )
        return g

    # self loops, problem precedence, problem precedence (reversed)
    # resource constraints (already bidir)
    # resource priority , resource priority (reversed)

    # def __init__(self, graphs, glist=False):
    #     self.graphs = graphs
    #     self.glist = glist

    def __init__(
        self,
        graph_observation,
        conflicts="att",
        #        add_self_loops=True,
        device=None,
        do_batch=True,
        compute_laplacian_pe=False,
        laplacian_pe_cache=None,
        n_laplacian_eigv=50,
        bidir=True,
        factored_rp=False,
        add_rp_edges="all",
        max_n_resources=-1,
        rwpe_k=0,
        rwpe_cache=None,
        rewire_internal=False,
    ):
        if not isinstance(graph_observation, list):
            graph_observation = [graph_observation]
        # batching on CPU for performance reasons...
        n_nodes = torch.tensor([g.num_nodes() for g in graph_observation])

        # n_pr_edges = torch.tensor([g.n_edges(etype="pr") for g in graph_observation])
        # pr_edges = gym_observation["pr_edges"].long().to(torch.device("cpu"))
        # if add_rp_edges != "none":
        #     n_rp_edges = gym_observation["n_rp_edges"].long().to(torch.device("cpu"))
        #     rp_edges = gym_observation["rp_edges"].long().to(torch.device("cpu"))
        #     rp_att = gym_observation["rp_att"].to(torch.device("cpu"))

        # orig_feat = gym_observation["features"]
        # .to(torch.device("cpu"))

        if do_batch:
            batch_num_nodes = []
            batch_num_edges = {
                "prec": [],
                "rprec": [],
                "rp": [],
                "rrp": [],
                "rc": [],
            }

        self.graphs = []

        for g in graph_observation:
            if rewire_internal:
                g = AgentGraphObservation.rewire_internal(
                    g,
                    conflicts,
                    bidir,
                    compute_laplacian_pe,
                    laplacian_pe_cache,
                    rwpe_k,
                    rwpe_cache,
                )

            if do_batch:
                batch_num_nodes.append(g.num_nodes())
                batch_num_edges["prec"].append(g.num_edges(etype="prec"))
                batch_num_edges["rprec"].append(g.num_edges(etype="rprec"))
                batch_num_edges["rp"].append(g.num_edges(etype="rp"))
                batch_num_edges["rrp"].append(g.num_edges(etype="rrp"))
                batch_num_edges["rc"].append(g.num_edges(etype="rc"))

            self.graphs.append(g)

        if do_batch:
            self.glist = False
            self.graphs = dgl.batch(self.graphs)
            self.graphs.set_batch_num_nodes(torch.tensor(batch_num_nodes))
            self.graphs.set_batch_num_edges(batch_num_edges)

        else:
            self.glist = True

    @classmethod
    def rwpe(cls, g, k, rwpe_cache):
        if rwpe_cache is not None:
            edges = g.edges(order="srcdst")
            key = (
                g.num_nodes(),
                *(edges[0].tolist() + edges[1].tolist()),
            )
            if key in rwpe_cache:
                return rwpe_cache[key]
            else:
                pe = dgl.random_walk_pe(g, k)
                rwpe_cache[key] = pe
                return pe
        return dgl.random_walk_pe(g, k)
