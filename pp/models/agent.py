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
import pickle

from generic.agent import Agent
from generic.models.gnn_mp import GnnMP

# from .gnn_tokengt import GnnTokenGT
from generic.mlp import MLP
from pp.graph.graph_factory import GraphFactory
from functools import partial
import numpy as np

# from .agent_observation import AgentObservation
import copy
from tqdm.contrib.concurrent import process_map
from generic.agent_obs import AgentObservation, AgentObservationBatch
from pp.models.node_embedders import (
    SimpleNodeEmbedder,
    SimpleNodeEmbedderNonChrono,
    PoolNodeEmbedder,
    MASimpleNodeEmbedder,
)
from pp.models.edge_embedder import EdgeEmbedder
from pp.models.path_action import PathAction
from pp.graph.pyg_graph import PYGGraph


class Agent(Agent):
    def __init__(
        self,
        env_specification,
        gnn=None,
        value_net=None,
        action_net=None,
        agent_specification=None,
        do_compile=True,
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

        self.do_compile = do_compile
        pp_agent_types = agent_specification.agent_types
        if pp_agent_types is None:
            pp_agent_types = [0]
        self.nonchrono = agent_specification.nonchrono
        if self.nonchrono == "path":
            self.action_dim = env_specification.max_n_nodes
        self.pp_agent_types = list(pp_agent_types)
        self.num_agents = len(self.pp_agent_types)
        self.graph_pooling = agent_specification.graph_pooling
        self.graphobs = True
        self.env_specification = env_specification
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
            return

        multi_agent = self.pp_agent_types is not None and len(self.pp_agent_types) > 1

        if self.nonchrono is not None:
            node_embedder_type = SimpleNodeEmbedderNonChrono
        else:
            node_embedder_type = (
                MASimpleNodeEmbedder if multi_agent else SimpleNodeEmbedder
            )
        node_embedders = {
            "n": {
                "class": node_embedder_type,
                "kwargs": {
                    "n_layers": 2,
                    "activation": "gelu",
                    "lappe": self.agent_specification.lappe,
                    "rwpe": self.agent_specification.rwpe,
                },
            },
        }
        if self.nonchrono is not None:
            node_embedders["n"]["kwargs"]["nonchrono"] = self.nonchrono
        if (
            self.graph_pooling in ["learn", "learninv"]
            and not self.agent_specification.hierarchical
        ):
            node_embedders["poolnode"] = {"class": PoolNodeEmbedder, "kwargs": {}}

        edge_embedders = {
            ("n", "free", "n"): {
                "class": EdgeEmbedder,
                "kwargs": {},
            },
            ("n", "wall", "n"): {
                "class": EdgeEmbedder,
                "kwargs": {},
            },
            ("n", "self_n", "n"): {
                "class": EdgeEmbedder,
                "kwargs": {},
            },
            ("poolnode", "selfpool", "poolnode"): {
                "class": EdgeEmbedder,
                "kwargs": {},
            },
            ("n", "path_graph", "n"): {
                "class": EdgeEmbedder,
                "kwargs": {},
            },
            ("n", "path_graph_inv", "n"): {
                "class": EdgeEmbedder,
                "kwargs": {},
            },
        }
        if agent_specification.graph_pooling in ["learn", "learninv"]:
            edge_embedders[("n", "pool", "poolnode")] = {
                "class": EdgeEmbedder,
                "kwargs": {},
            }
        if agent_specification.graph_pooling == "learninv":
            edge_embedders[("poolnode", "rpool", "n")] = {
                "class": EdgeEmbedder,
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
            do_compile=do_compile,
        )

        self.init_heads()

    @classmethod
    def load(cls, path, max_n_modes=None):
        """Loading an agent corresponds to loading his model and a few args to specify how the model is working"""
        save_data = torch.load(path + "agent.pkl", weights_only=False)
        agent_specification = save_data["agent_specification"]
        env_specification = save_data["env_specification"]
        if max_n_modes is not None:
            env_specification.max_n_modes = max_n_modes
            env_specification.max_n_nodes = max_n_modes
            env_specification.max_n_jobs = max_n_modes
        if agent_specification.agent_types is None:
            agent_specification.agent_types = save_data.get("pp_agent_types")
        agent = cls(env_specification, agent_specification=agent_specification)

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
        if self.do_compile:
            self.value_net = torch.compile(self.value_net, dynamic=True)
        self.action_nets = torch.nn.ModuleList()
        for _ in range(self.num_agents):
            if self.nonchrono == "path":
                head = PathAction(
                    len(self.agent_specification.net_arch["pi"]),
                    self.gnn.features_dim,
                    self.agent_specification.net_arch["pi"][0],
                    2,
                    False,
                    self.agent_specification.activation_fn,
                )
            else:
                head = MLP(
                    len(self.agent_specification.net_arch["pi"]),
                    self.gnn.features_dim,
                    self.agent_specification.net_arch["pi"][0],
                    1,
                    False,
                    self.agent_specification.activation_fn,
                )

            if self.do_compile:
                head = torch.compile(head, dynamic=True)
            self.action_nets.append(head)
        if self.num_agents == 1:
            self.action_net = self.action_nets[0]

        self.value_net.to(device)
        for head in self.action_nets:
            head.to(device)

    def obs_as_tensor_add_batch_dim(self, obs):
        cobs = obs.clone()
        return AgentObservation(
            cobs,
            self.rewire_params,
        )

    def obs_as_tensor(self, obs):
        # create agentObs from graph from output data from env
        return [
            AgentObservation(
                o,
                self.rewire_params,
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
            return [AgentObservation.load(b_obs[i], PYGGraph) for i in mb_ind]
        return list(b_obs[i] for i in mb_ind)

    def get_action_and_value_nonchrono2(self, nfeats, bactions, deterministic):
        actions = []
        logprobs = []
        entropies = []
        for idx, head in enumerate(self.action_nets):
            logits = head(nfeats)
            distribs = torch.distributions.normal.Normal(logits[..., 0], logits[..., 1])
            if bactions is None:
                if deterministic is False:
                    unclipped_actions = distribs.sample()
                else:
                    unclipped_actions = distribs.mean
            else:
                unclipped_actions = bactions[..., idx]

            # clipped_actions = torch.clamp(unclipped_actions, min=-1.0, max=1.0)
            # logprobs.append(distribs.log_prob(clipped_actions))
            logprobs.append(distribs.log_prob(unclipped_actions))
            entropies.append(distribs.entropy())
            # if bactions is None:
            #     actions.append(clipped_actions)
            if bactions is None:
                actions.append(unclipped_actions)
        if bactions is None:
            return (
                torch.stack(actions, dim=-1),
                torch.stack(logprobs, dim=-1),
                torch.stack(entropies, dim=-1),
            )
        return (
            bactions,
            torch.stack(logprobs, dim=-1),
            torch.stack(entropies, dim=-1),
        )

    def get_action_and_value(
        self, x, action=None, action_masks=None, deterministic=False
    ):
        if self.nonchrono != "path" and self.num_agents == 1:
            return super().get_action_and_value(x, action, action_masks, deterministic)

        node_features, graph_embedding = self.gnn(x)
        value = self.value_net(graph_embedding)

        if self.nonchrono == "path":
            actions, logprobs, entropies = self.get_action_and_value_nonchrono2(
                node_features, action, deterministic
            )
            return actions, logprobs, entropies, value, None
