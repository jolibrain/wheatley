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

from models.gnn_dgl import GnnDGL
from models.gnn_tokengt import GnnTokenGT
from problem.problem_description import ProblemDescription
from models.mlp import MLP
from torch.distributions.categorical import Categorical
from functools import partial
import numpy as np


class PSPAgent(torch.nn.Module):
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
            env_specification, gnn, value_net, action_net, agent_specification
        )
        # User must provide an agent_specification or a model at least.
        if agent_specification is None or env_specification is None:
            raise Exception(
                "Please provide an agent_specification to create a new Agent"
            )

        self.env_specification = env_specification
        self.agent_specification = agent_specification

        # If a model is provided, we simply load the existing model.
        if gnn is not None and value_net is not None and action_net is not None:
            self.gnn = gnn
            self.value_net = value_net
            self.action_net = action_net
            return

        if self.agent_specification.fe_type == "dgl":
            self.gnn = GnnDGL(
                input_dim_features_extractor=env_specification.n_features,
                gconv_type=agent_specification.gconv_type,
                graph_pooling=agent_specification.graph_pooling,
                graph_has_relu=agent_specification.graph_has_relu,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_machines=env_specification.max_n_machines,
                n_mlp_layers_features_extractor=agent_specification.n_mlp_layers_features_extractor,
                activation_features_extractor=agent_specification.activation_fn_graph,
                n_layers_features_extractor=agent_specification.n_layers_features_extractor,
                hidden_dim_features_extractor=agent_specification.hidden_dim_features_extractor,
                n_attention_heads=agent_specification.n_attention_heads,
                residual=agent_specification.residual_gnn,
                normalize=agent_specification.normalize_gnn,
                conflicts=agent_specification.conflicts,
            )
        elif self.agent_specification.fe_type == "tokengt":
            self.gnn = GnnTokenGT(
                input_dim_features_extractor=env_specification.n_features,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_machines=env_specification.max_n_machines,
                conflicts=agent_specification.conflicts,
                encoder_layers=agent_specification.n_layers_features_extractor,
                encoder_embed_dim=agent_specification.hidden_dim_features_extractor,
                encoder_ffn_embed_dim=agent_specification.hidden_dim_features_extractor,
                encoder_attention_heads=agent_specification.n_attention_heads,
                activation_fn=agent_specification.activation_fn_graph,
                lap_node_id=True,
                lap_node_id_k=agent_specification.lap_node_id_k,
                lap_node_id_sign_flip=True,
                type_id=True,
                transformer_flavor=agent_specification.transformer_flavor,
                layer_pooling=agent_specification.layer_pooling,
                dropout=agent_specification.dropout,
                attention_dropout=agent_specification.dropout,
                act_dropout=agent_specification.dropout,
                cache_lap_node_id=agent_specification.cache_lap_node_id,
            )
        else:
            print("unknown fe_type: ", agent_specification.fe_type)

        self.value_net = MLP(
            len(self.agent_specification.net_arch["vf"]),
            self.gnn.features_dim // 2,
            self.agent_specification.net_arch["vf"][0],
            1,
            False,
            self.agent_specification.activation_fn,
        )
        # # action
        self.action_net = MLP(
            len(self.agent_specification.net_arch["pi"]),
            self.gnn.features_dim,
            self.agent_specification.net_arch["pi"][0],
            1,
            False,
            self.agent_specification.activation_fn,
        )
        # usually ppo use gain = np.sqrt(2) here
        # best so far below
        self.gnn.apply(partial(self.init_weights, gain=1.0, zero_bias=False))
        # usually ppo use gain = 0.01 here
        self.action_net.apply(partial(self.init_weights, gain=1.0, zero_bias=True))
        # usually ppo use gain = 1 here
        self.value_net.apply(partial(self.init_weights, gain=1.0, zero_bias=True))

    @staticmethod
    def init_weights(module, gain=1, zero_bias=True) -> None:
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None and zero_bias:
                module.bias.data.fill_(0.0)

    def save(self, path):
        """Saving an agent corresponds to saving his model and a few args to specify how the model is working"""
        device = next(self.gnn.parameters()).device
        self.to(torch.device("cpu"))
        torch.save(
            {
                "env_specification": self.env_specification,
                "agent_specification": self.agent_specification,
                "gnn": self.gnn.state_dict(),
                "value_net": self.value_net.state_dict(),
                "action_net": self.action_net.state_dict(),
            },
            path,
        )
        self.to(device)

    @classmethod
    def load(cls, path):
        """Loading an agent corresponds to loading his model and a few args to specify how the model is working"""
        save_data = torch.load(path + "agent.pkl")
        agent_specification = save_data["agent_specification"]
        env_specification = save_data["env_specification"]
        if agent_specification.fe_type == "dgl":
            gnn = GnnDGL(
                input_dim_features_extractor=env_specification.n_features,
                gconv_type=agent_specification.gconv_type,
                graph_pooling=agent_specification.graph_pooling,
                graph_has_relu=agent_specification.graph_has_relu,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_machines=env_specification.max_n_machines,
                n_mlp_layers_features_extractor=agent_specification.n_mlp_layers_features_extractor,
                activation_features_extractor=agent_specification.activation_fn_graph,
                n_layers_features_extractor=agent_specification.n_layers_features_extractor,
                hidden_dim_features_extractor=agent_specification.hidden_dim_features_extractor,
                n_attention_heads=agent_specification.n_attention_heads,
                reverse_adj=agent_specification.reverse_adj,
                residual=agent_specification.residual_gnn,
                normalize=agent_specification.normalize_gnn,
                conflicts=agent_specification.conflicts,
            )
        elif agent_specification.fe_type == "tokengt":
            gnn = GnnTokenGT(
                input_dim_features_extractor=env_specification.n_features,
                device=agent_specification.device,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_machines=env_specification.max_n_machines,
                conflicts=agent_specification.conflicts,
                encoder_layers=agent_specification.n_layers_features_extractor,
                encoder_embed_dim=agent_specification.hidden_dim_features_extractor,
                encoder_ffn_embed_dim=agent_specification.hidden_dim_features_extractor,
                encoder_attention_heads=agent_specification.n_attention_heads,
                activation_fn=agent_specification.activation_fn_graph,
                lap_node_id=True,
                lap_node_id_k=agent_specification.lap_node_id_k,
                lap_node_id_sign_flip=True,
                type_id=True,
                transformer_flavor=agent_specification.transformer_flavor,
                layer_pooling=agent_specification.layer_pooling,
                dropout=agent_specification.dropout,
                attention_dropout=agent_specification.dropout,
                act_dropout=agent_specification.dropout,
                cache_lap_node_id=agent_specification.cache_lap_node_id,
            )
        value_net = MLP(
            len(agent_specification.net_arch["vf"]),
            gnn.features_dim // 2,
            agent_specification.net_arch["vf"][0],
            1,
            False,
            agent_specification.activation_fn,
        )

        # # action
        action_net = MLP(
            len(agent_specification.net_arch["pi"]),
            gnn.features_dim,
            agent_specification.net_arch["pi"][0],
            1,
            False,
            agent_specification.activation_fn,
        )

        agent = cls(env_specification, gnn, value_net, action_net, agent_specification)
        # constructors init weight!!!
        agent.gnn.load_state_dict(save_data["gnn"])
        agent.action_net.load_state_dict(save_data["action_net"])
        agent.value_net.load_state_dict(save_data["value_net"])
        return agent
