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


import torch
import pickle

from generic.agent import Agent
from generic.models.gnn_mp import GnnMP

# from .gnn_tokengt import GnnTokenGT
from generic.mlp import MLP
from functools import partial
import numpy as np

import copy
from tqdm.contrib.concurrent import process_map
from generic.agent_obs import AgentObservation, AgentObservationBatch
from psp.models.node_embedders import (
    SimpleNodeEmbedder,
    PoolNodeEmbedder,
    ResNodeEmbedder,
)
from psp.models.edge_embedder import SimpleEdgeEmbedder, RPEdgeEmbedder, UseEdgeEmbedder
from psp.models.rewirer import PspRewirer


class Agent(Agent):
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
        "poolres": 12,
        "rpoolres": 13,
        "selfpool": 14,
        "selfres": 15,
    }

    def __init__(
        self,
        env_specification,
        gnn=None,
        value_net=None,
        action_net=None,
        agent_specification=None,
    ):
        """
        There are 2 ways to init an Agent:
         - Either provide a valid env_specification and agent_specification
         - Or use the load method, to load an already saved Agent
        """
        super().__init__(
            env_specification,
            gnn,
            value_net,
            action_net,
            agent_specification,
        )

        self.graph_pooling = agent_specification.graph_pooling
        self.env_specification = env_specification
        self.psp_rewirer = PspRewirer()
        self.rewire_params = {
            "bidir": self.agent_specification.bidir,
            "graph_pooling": None
            if self.agent_specification.hierarchical
            else agent_specification.graph_pooling,
            "vnoding": False,
            "self_loops": self.agent_specification.self_loops,
        }
        # If a model is provided, we simply load the existing model.
        if gnn is not None and value_net is not None and action_net is not None:
            self.gnn = gnn
            self.value_net = value_net
            if isinstance(action_net, torch.nn.ModuleList):
                self.action_nets = action_net
            else:
                self.action_nets = torch.nn.ModuleList([action_net])
            # self.action_net = self.action_nets[0]
            return
        node_embedder_type = SimpleNodeEmbedder
        node_embedders = {
            "n": {
                "class": node_embedder_type,
                "kwargs": {
                    # "n_layers": self.agent_specification.n_layers_features_extractor,
                    "n_layers": 2,
                    # "activation": self.agent_specification.gconv_activation,
                    "activation": "gelu",
                    "lappe": self.agent_specification.lappe,
                    "rwpe": self.agent_specification.rwpe,
                },
            },
        }
        if (
            self.graph_pooling in ["learn", "learninv"]
            and not self.agent_specification.hierarchical
        ):
            node_embedders["poolnode"] = {"class": PoolNodeEmbedder, "kwargs": {}}
        node_embedders["resource"] = {
            "class": ResNodeEmbedder,
            "kwargs": {"max_n_resources": self.env_specification.max_n_resources},
        }

        edge_embedders = {
            ("n", "prec", "n"): {
                "class": SimpleEdgeEmbedder,
                "kwargs": {},
            },
            ("n", "rprec", "n"): {
                "class": SimpleEdgeEmbedder,
                "kwargs": {},
            },
            ("n", "rp", "n"): {
                "class": RPEdgeEmbedder,
                "kwargs": {
                    "n_layers": 1,
                    "activation": "gelu",
                },
            },
            ("n", "rrp", "n"): {
                "class": RPEdgeEmbedder,
                "kwargs": {
                    "n_layers": 1,
                    "activation": "gelu",
                },
            },
            ("n", "uses", "resource"): {
                "class": UseEdgeEmbedder,
                "kwargs": {
                    "n_layers": 1,
                    "activation": "gelu",
                },
            },
            ("resource", "ruses", "n"): {
                "class": UseEdgeEmbedder,
                "kwargs": {
                    "n_layers": 1,
                    "activation": "gelu",
                },
            },
            ("n", "self_n", "n"): {
                "class": SimpleEdgeEmbedder,
                "kwargs": {},
            },
            ("resource", "self_resource", "resource"): {
                "class": SimpleEdgeEmbedder,
                "kwargs": {},
            },
            ("poolnode", "selfpool", "poolnode"): {
                "class": SimpleEdgeEmbedder,
                "kwargs": {},
            },
        }
        if agent_specification.graph_pooling in ["learn", "learninv"]:
            edge_embedders[("n", "pool", "poolnode")] = {
                "class": SimpleEdgeEmbedder,
                "kwargs": {},
            }
        if agent_specification.graph_pooling == "learninv":
            edge_embedders[("poolnode", "rpool", "n")] = {
                "class": SimpleEdgeEmbedder,
                "kwargs": {},
            }

        self.gnn = GnnMP(
            node_embedders,
            edge_embedders,
            graph_pooling=agent_specification.graph_pooling,
            max_n_nodes=env_specification.max_n_nodes,
            n_mlp_layers_features_extractor=agent_specification.n_mlp_layers_features_extractor,
            activation_features_extractor=agent_specification.activation_fn_graph,
            n_layers_features_extractor=agent_specification.n_layers_features_extractor,
            hidden_dim_features_extractor=agent_specification.hidden_dim_features_extractor,
            n_attention_heads=agent_specification.n_attention_heads,
            residual=agent_specification.residual_gnn,
            normalize=agent_specification.normalize_gnn,
            layer_pooling=agent_specification.layer_pooling,
            hierarchical=agent_specification.hierarchical,
            checkpoint=agent_specification.checkpoint
            if hasattr(agent_specification, "checkpoint")
            else 1,
            gconv_activation=agent_specification.gconv_activation,
            shared_layers=agent_specification.shared_conv,
            g2=agent_specification.g2,
            num_node_types=3,  # tasks, poolnode, resnode
        )

        self.init_heads()

    @classmethod
    def load(cls, path, max_n_modes=None, pp_agent_types=None, nonchrono=None):
        """Loading an agent corresponds to loading his model and a few args to specify how the model is working"""
        save_data = torch.load(path + "agent.pkl", weights_only=False)
        agent_specification = save_data["agent_specification"]
        env_specification = save_data["env_specification"]
        if max_n_modes is not None:
            env_specification.max_n_modes = max_n_modes
            env_specification.max_n_nodes = max_n_modes
            env_specification.max_n_jobs = max_n_modes
        if pp_agent_types is None:
            pp_agent_types = save_data.get("pp_agent_types")
        agent = cls(
            env_specification,
            agent_specification=agent_specification,
            pp_agent_types=pp_agent_types,
            nonchrono=nonchrono,
        )

        # constructors init weight!!!
        agent.gnn.load_state_dict(save_data["gnn"])
        agent.value_net.load_state_dict(save_data["value_net"])
        action_state = save_data["action_net"]
        if isinstance(action_state, list):
            for head, state in zip(agent.action_nets, action_state):
                head.load_state_dict(state)
            if len(action_state) < len(agent.action_nets):
                fallback_state = action_state[-1]
                for head in agent.action_nets[len(action_state) :]:
                    head.load_state_dict(fallback_state)
        else:
            for head in agent.action_nets:
                head.load_state_dict(action_state)
        agent.action_net = agent.action_nets[0]
        return agent

    def init_heads(self):
        """Initialize new heads, removing old heads if existing."""
        if hasattr(self, "value_net") and self.value_net is not None:
            device = next(self.value_net.parameters())
        else:
            device = "cpu"

        if self.agent_specification.two_hot is not None:
            value_dim = len(self.B)
        elif self.agent_specification.hl_gauss is not None:
            value_dim = len(self.B) - 1
        else:
            value_dim = 1

        self.value_net = MLP(
            len(self.agent_specification.net_arch["vf"]),
            self.gnn.features_dim,
            self.agent_specification.net_arch["vf"][0],
            value_dim,
            False,
            self.agent_specification.activation_fn,
            #            "gelu",
        )
        self.value_net = torch.compile(self.value_net, dynamic=True)
        self.action_net = MLP(
            len(self.agent_specification.net_arch["pi"]),
            self.gnn.features_dim,
            self.agent_specification.net_arch["pi"][0],
            1,
            False,
            self.agent_specification.activation_fn,
            #            "gelu",
        )
        self.action_net = torch.compile(self.action_net, dynamic=True)

    def preprocess(self, obs):
        # works for batches or simple graphs
        # do external rewiring + homogeneous_edges
        observation = AgentObservationBatch.from_aos(obs)

        (
            g,
            batch_size,
            total_num_nodes,
            total_num_edges,
            num_nodes,
            num_edges,
            nodesid,
            edgesid,
        ) = observation.homogeneous()

        return (
            g,
            batch_size,
            total_num_nodes,
            total_num_edges,
            num_nodes,
            num_edges,
            nodesid,
            edgesid,
        )

    def obs_as_tensor_add_batch_dim(self, obs):
        cobs = obs.clone()
        return AgentObservation(
            cobs,
            self.rewire_params,
            self.psp_rewirer,
        )

    def obs_as_tensor(self, obs):
        # create agentObs from graph from output data from env
        return [
            AgentObservation(
                o,
                self.rewire_params,
                self.psp_rewirer,
            )
            for o in obs
        ]

    def rebatch_obs(self, obs):
        # we need to flatten a list of list into a single list
        if isinstance(obs[0], str):
            return obs
        return sum(obs, [])

    def get_obs(self, b_obs, mb_ind):
        if isinstance(b_obs[0], str):
            # return [dgl.load_graphs(b_obs[i])[0][0] for i in mb_ind]
            # bobsi = [b_obs[i] for i in mb_ind]
            # return process_map(
            #     GraphFactory.load, bobsi, max_workers=16, chunksize=1, disable=True
            # )
            return [
                # GraphFactory.load(b_obs[i], self.env_specification.pyg) for i in mb_ind
                AgentObservation.load(b_obs[i])
                for i in mb_ind
            ]
        return list(b_obs[i] for i in mb_ind)
