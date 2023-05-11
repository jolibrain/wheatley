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
from models.tokengt.utils import get_laplacian_pe_simple
from .utils import (
    compute_resources_graph_torch,
)
import dgl
import time


class PSPAgentObservation:
    edgeType = {"self": 0, "prec": 1, "rprec": 2, "rc": 3, "rp": 4, "rrp": 5}
    # self loops, problem precedence, problem precedence (reversed)
    # resource constraints (already bidir)
    # resource priority , resource priority (reversed)

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
        type0 = [cls.edgeType["prec"]] * n_edges
        if bidir:
            type1 = [cls.edgeType["rprec"]] * n_edges

            gnew = dgl.graph(
                (torch.cat([edges0, edges1]), torch.cat([edges1, edges0])),
                num_nodes=nnodes,
            )
        else:
            gnew = dgl.graph((edges0, edges1), num_nodes=nnodes)

        gnew.ndata["feat"] = feats
        if bidir:
            type0.extend(type1)
            nn_edges = n_edges * 2
        else:
            nn_edges = n_edges
        gnew.edata["type"] = torch.LongTensor(type0)
        gnew.edata["rid"] = torch.zeros((nn_edges), dtype=torch.int)
        gnew.edata["att_rp"] = torch.zeros((nn_edges, 3), dtype=torch.float32)
        gnew.edata["att_rc"] = torch.zeros((nn_edges, 2), dtype=torch.float32)
        return gnew

    @classmethod
    def get_machine_id(cls, machine_one_hot):
        # print("machine_one_hot", machine_one_hot)

        return torch.max(machine_one_hot, dim=1)

    @classmethod
    def add_edges(cls, g, t, edges, rid, att):
        t_type = torch.LongTensor([cls.edgeType[t]] * edges.shape[1])
        if t == "rrp":
            tt = "rp"
        else:
            tt = t
        if tt == "rp":
            # offset for timetype to make it different from fill value (which is zero)
            att[:, -1] = att[:, -1] + 1
        g.add_edges(
            edges[0],
            edges[1],
            data={"type": t_type, "rid": rid.int() + 1, "att_" + tt: att},
        )
        return g

    @classmethod
    def rebatch_obs(cls, obs):
        rebatched_obs = {}
        max_nnodes = 0
        max_n_pr_edges = 0
        max_n_rc_edges = 0
        max_n_rp_edges = 0
        num_steps = len(obs)
        for _obs in obs:
            mnn = _obs["n_nodes"].max().item()
            if mnn > max_nnodes:
                max_nnodes = mnn
            mnpre = _obs["n_pr_edges"].max().item()
            if mnpre > max_n_pr_edges:
                max_n_pr_edges = mnpre
            mnrpe = _obs["n_rp_edges"].max().item()
            if mnrpe > max_n_rp_edges:
                max_n_rp_edges = mnrpe
            if "rc_edges" in _obs:
                mnrce = _obs["n_rc_edges"].max().item()
                if mrce > max_n_rc_edges:
                    max_n_rc_edges = mnrce
        max_nnodes = int(max_nnodes)
        max_n_pr_edges = int(max_n_pr_edges)
        max_n_rp_edges = int(max_n_rp_edges)
        max_n_rc_edges = int(max_n_rc_edges)

        for key in obs[0]:
            if key == "features":
                s = (max_nnodes, obs[0][key].shape[-1])
                rebatched_obs[key] = torch.stack(
                    [
                        torch.nn.functional.pad(
                            obs[j][key],
                            (0, 0, 0, max_nnodes - obs[j][key].shape[-2]),
                        )
                        for j in range(num_steps)
                    ]
                ).reshape(torch.Size((-1,)) + s)
            elif key == "pr_edges":
                s = (obs[0][key].shape[-2], max_n_pr_edges)
                rebatched_obs[key] = torch.stack(
                    [
                        torch.nn.functional.pad(
                            obs[j][key],
                            (0, max_n_pr_edges - obs[j][key].shape[-1]),
                        )
                        for j in range(num_steps)
                    ]
                ).reshape(torch.Size((-1,)) + s)
            elif key in ["rp_edges"]:
                s = (obs[0][key].shape[-2], max_n_rp_edges)
                l = [
                    torch.nn.functional.pad(
                        obs[j][key],
                        (0, max_n_rp_edges - obs[j][key].shape[-1]),
                    )
                    for j in range(num_steps)
                ]
                rebatched_obs[key] = torch.stack(l).reshape(torch.Size((-1,)) + s)
            elif key in ["rp_att"]:
                num_feat = obs[0][key].shape[-1]
                l = [
                    torch.nn.functional.pad(
                        obs[j][key],
                        (0, 0, 0, max_n_rp_edges - obs[j][key].shape[-2]),
                    )
                    for j in range(num_steps)
                ]
                rebatched_obs[key] = torch.stack(l).reshape(
                    (-1, max_n_rp_edges, num_feat)
                )

            elif key in ["rc_edges"]:
                s = (obs[0][key].shape[-2], max_n_rc_edges)
                rebatched_obs[key] = torch.stack(
                    [
                        torch.nn.functional.pad(
                            obs[j][key],
                            (
                                0,
                                max_n_rc_edges - obs[j][key].shape[-1],
                            ),
                        )
                        for j in range(num_steps)
                    ]
                ).reshape(torch.Size((-1,)) + s)
            elif key == "rc_att":
                num_feat = obs[0][key].shape[-1]
                l = [
                    torch.nn.functional.pad(
                        obs[j][key],
                        (0, 0, 0, max_n_rc_edges - obs[j][key].shape[-2]),
                    )
                    for j in range(num_steps)
                ]
                rebatched_obs[key] = torch.stack(l).reshape(
                    (-1, max_n_rc_edges, num_feat)
                )

            else:
                s = obs[0][key].shape
                rebatched_obs[key] = torch.stack(
                    [obs[j][key] for j in range(num_steps)]
                ).reshape(torch.Size((-1,)) + s[1:])
        return rebatched_obs

    @classmethod
    def np_to_torch(cls, obs):
        if isinstance(obs, np.ndarray):
            return torch.tensor(obs)
        elif isinstance(obs, dict):
            has_batch_dim = obs["features"].ndim == 3
            max_n_nodes = np.max(obs["n_nodes"])
            max_pr_edges = np.max(obs["n_pr_edges"])
            max_rp_edges = np.max(obs["n_rp_edges"])
            if "rc_edges" in obs:
                max_rc_edges = np.max(obs["n_rc_edges"])
            newobs = {}
            if has_batch_dim:
                for key, _obs in obs.items():
                    if key == "features":
                        newobs[key] = torch.tensor(
                            _obs[:, :max_n_nodes, :], dtype=torch.float
                        )
                    elif key == "pr_edges":
                        newobs[key] = torch.tensor(
                            _obs[:, :, :max_pr_edges], dtype=torch.long
                        )
                    elif key == "rp_edges":
                        newobs[key] = torch.tensor(
                            _obs[:, :, :max_rp_edges], dtype=torch.long
                        )
                    elif key == "rc_edges":
                        newobs[key] = torch.tensor(
                            _obs[:, :, :max_rc_edges], dtype=torch.long
                        )
                    elif key == "rc_att":
                        newobs[key] = torch.tensor(
                            _obs[:, :max_rc_edges, :], dtype=torch.float
                        )
                    elif key == "rp_att":
                        newobs[key] = torch.tensor(
                            _obs[:, :max_rp_edges, :], dtype=torch.float
                        )
                    else:
                        newobs[key] = torch.tensor(_obs, dtype=torch.long)
            else:
                for key, _obs in obs.items():
                    if key == "features":
                        newobs[key] = torch.tensor(
                            _obs[:max_n_nodes, :], dtype=torch.float
                        ).unsqueeze(0)
                    elif key == "pr_edges":
                        newobs[key] = torch.tensor(
                            _obs[:, :max_pr_edges], dtype=torch.long
                        ).unsqueeze(0)
                    elif key == "rp_edges":
                        newobs[key] = torch.tensor(
                            _obs[:, :max_rp_edges], dtype=torch.long
                        ).unsqueeze(0)
                    elif key == "rc_edges":
                        newobs[key] = torch.tensor(
                            _obs[:, :max_rc_edges], dtype=torch.long
                        ).unsqueeze(0)
                    elif key == "rc_att":
                        newobs[key] = torch.tensor(
                            _obs[:max_rc_edges, :], dtype=torch.float
                        ).unsqueeze(0)
                    elif key == "rp_att":
                        newobs[key] = torch.tensor(
                            _obs[:max_rp_edges, :], dtype=torch.float
                        ).unsqueeze(0)
                    else:  # n_ stuff
                        newobs[key] = torch.tensor([_obs], dtype=torch.long)

            return newobs
        else:
            raise Exception(f"Unrecognized type of observation {type(obs)}")

    @classmethod
    def from_gym_observation(
        cls,
        gym_observation,
        conflicts="att",
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

        n_pr_edges = gym_observation["n_pr_edges"].long().to(torch.device("cpu"))
        pr_edges = gym_observation["pr_edges"].long().to(torch.device("cpu"))
        n_rp_edges = gym_observation["n_rp_edges"].long().to(torch.device("cpu"))
        rp_edges = gym_observation["rp_edges"].long().to(torch.device("cpu"))
        rp_att = gym_observation["rp_att"].to(torch.device("cpu"))

        orig_feat = gym_observation["features"]
        # .to(torch.device("cpu"))

        if conflicts != "clique":
            orig_feat = orig_feat.to(torch.device("cpu"))
        else:
            if "rc_edges" in gym_observation:  # precomputed cliques
                n_rc_edges = (
                    gym_observation["n_rc_edges"].long().to(torch.device("cpu"))
                )

                rc_edges = gym_observation["rc_edges"].long().to(torch.device("cpu"))
                rc_att = gym_observation["rc_att"].to(torch.device("cpu"))
                orig_feat = orig_feat.to(torch.device("cpu"))
            else:  # compute cliques
                rc_edges = []
                rc_att = []
                all_nce = []

                for i in range(orig_feat.shape[0]):
                    one_rc_edges, rid, rval, rvalr = compute_resources_graph_torch(
                        orig_feat[i, :, 9:]
                    )
                    rc_edges.append(one_rc_edges)
                    att = torch.stack([rid, rval, rvalr]).t()
                    rc_att.append(att)
                    nce = torch.LongTensor([one_rc_edges.shape[1]])
                    all_nce.append(nce)

                n_rc_edges = torch.cat(all_nce)
                orig_feat = orig_feat.to(torch.device("cpu"))

        graphs = []
        if do_batch:
            batch_num_nodes = []
            batch_num_edges = []

        for i, nnodes in enumerate(n_nodes):
            features = orig_feat[i, :nnodes, :]
            gnew = cls.build_graph(
                n_pr_edges[i],
                pr_edges[i, :, : n_pr_edges[i].item()],
                nnodes.item(),
                orig_feat[i, : nnodes.item(), :],
                bidir,
            )

            # resource pririty edges
            if rp_edges.nelement() != 0:
                gnew = PSPAgentObservation.add_edges(
                    gnew,
                    "rp",
                    rp_edges[i][:, : n_rp_edges[i].item()],
                    rp_att[i][: n_rp_edges[i].item(), 0],
                    rp_att[i][: n_rp_edges[i].item(), 1:],
                )
                rpe = torch.empty((2, n_rp_edges[i].item()), dtype=rp_edges.dtype)
                rpe[0] = rp_edges[i][1, : n_rp_edges[i].item()]
                rpe[1] = rp_edges[i][0, : n_rp_edges[i].item()]
                gnew = PSPAgentObservation.add_edges(
                    gnew,
                    "rrp",
                    rpe,
                    rp_att[i][: n_rp_edges[i].item(), 0],
                    rp_att[i][: n_rp_edges[i].item(), 1:],
                )

            # resource conflicts edges
            if conflicts == "clique":
                gnew = PSPAgentObservation.add_edges(
                    gnew,
                    "rc",
                    rc_edges[i][:, : n_rc_edges[i].item()],
                    rc_att[i][: n_rc_edges[i].item(), 0],
                    rc_att[i][: n_rc_edges[i].item(), 1:],
                )

            # TODO : add job_id edges, ie link modes that are for same job

            if add_self_loops:
                gnew = dgl.add_self_loop(
                    gnew, edge_feat_names=["type"], fill_data=cls.edgeType["self"]
                )
            if compute_laplacian_pe:
                gnew.ndata["laplacian_pe"] = get_laplacian_pe_simple(
                    gnew, laplacian_pe_cache, n_laplacian_eigv
                )
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
