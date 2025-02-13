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
import pickle

from generic.agent import Agent
from .gnn_mp import GnnMP

# from .gnn_tokengt import GnnTokenGT
from generic.mlp import MLP
from psp.graph.graph_factory import GraphFactory
from functools import partial
import numpy as np

# from .agent_observation import AgentObservation
from .agent_graph_observation import AgentGraphObservation
import copy
from tqdm.contrib.concurrent import process_map
from psp.models.agent_graph_observation import AgentGraphObservation
from psp.models.rewiring import rewire, homogeneous_edges


class Agent(Agent):
    def __init__(
        self,
        env_specification,
        gnn=None,
        value_net=None,
        action_net=None,
        agent_specification=None,
        graphobs=True,
    ):
        """
        There are 2 ways to init an Agent:
         - Either provide a valid env_specification and agent_specification
         - Or use the load method, to load an already saved Agent
        """
        super().__init__(
            env_specification, gnn, value_net, action_net, agent_specification
        )

        self.graphobs = graphobs
        # if self.graphobs:
        self.obs_as_tensor_add_batch_dim = self._obs_as_tensor_add_batch_dim_graph
        self.obs_as_tensor = self._obs_as_tensor_graph
        self.rebatch_obs = self._rebatch_obs_graph
        self.get_obs = self._get_obs_graph
        # else:
        #     self.obs_as_tensor_add_batch_dim = self._obs_as_tensor_add_batch_dim
        #     self.obs_as_tensor = self._obs_as_tensor
        #     self.rebatch_obs = self._rebatch_obs
        #     self.get_obs = self._get_obs

        # If a model is provided, we simply load the existing model.
        if gnn is not None and value_net is not None and action_net is not None:
            self.gnn = gnn
            self.value_net = value_net
            self.action_net = action_net
            return

        if self.agent_specification.fe_type == "message_passing":
            if agent_specification.cache_rwpe:
                self.rwpe_cache = {}
            else:
                self.rwpe_cache = None

            self.gnn = GnnMP(
                input_dim_features_extractor=env_specification.n_features,
                graph_pooling=agent_specification.graph_pooling,
                max_n_nodes=env_specification.max_n_nodes,
                max_n_resources=env_specification.max_n_resources,
                n_mlp_layers_features_extractor=agent_specification.n_mlp_layers_features_extractor,
                activation_features_extractor=agent_specification.activation_fn_graph,
                n_layers_features_extractor=agent_specification.n_layers_features_extractor,
                hidden_dim_features_extractor=agent_specification.hidden_dim_features_extractor,
                n_attention_heads=agent_specification.n_attention_heads,
                residual=agent_specification.residual_gnn,
                normalize=agent_specification.normalize_gnn,
                conflicts=agent_specification.conflicts,
                edge_embedding_flavor=agent_specification.edge_embedding_flavor,
                layer_pooling=agent_specification.layer_pooling,
                factored_rp=env_specification.factored_rp,
                add_rp_edges=env_specification.add_rp_edges,
                add_self_loops=env_specification.remove_past_prec
                or env_specification.observe_subgraph,
                vnode=agent_specification.vnode,
                update_edge_features=agent_specification.update_edge_features,
                update_edge_features_pe=agent_specification.update_edge_features_pe,
                rwpe_k=agent_specification.rwpe_k,
                rwpe_h=agent_specification.rwpe_h,
                rwpe_cache=self.rwpe_cache,
                graphobs=graphobs,
                pyg=agent_specification.pyg,
                hierarchical=agent_specification.hierarchical,
                tokengt=agent_specification.tokengt,
                shared_conv=agent_specification.shared_conv,
                checkpoint=agent_specification.checkpoint
                if hasattr(agent_specification, "checkpoint")
                else 1,
                dual_net=agent_specification.dual_net,
            )
        # elif self.agent_specification.fe_type == "tokengt":
        #     self.gnn = GnnTokenGT(
        #         input_dim_features_extractor=env_specification.n_features,
        #         max_n_nodes=env_specification.max_n_nodes,
        #         max_n_resources=env_specification.max_n_resources,
        #         conflicts=agent_specification.conflicts,
        #         encoder_layers=agent_specification.n_layers_features_extractor,
        #         encoder_embed_dim=agent_specification.hidden_dim_features_extractor,
        #         encoder_ffn_embed_dim=agent_specification.hidden_dim_features_extractor,
        #         encoder_attention_heads=agent_specification.n_attention_heads,
        #         activation_fn=agent_specification.activation_fn_graph,
        #         lap_node_id=True,
        #         lap_node_id_k=agent_specification.lap_node_id_k,
        #         lap_node_id_sign_flip=True,
        #         type_id=True,
        #         transformer_flavor=agent_specification.transformer_flavor,
        #         layer_pooling=agent_specification.layer_pooling,
        #         dropout=agent_specification.dropout,
        #         attention_dropout=agent_specification.dropout,
        #         act_dropout=agent_specification.dropout,
        #         cache_lap_node_id=agent_specification.cache_lap_node_id,
        #         performer_nb_features=agent_specification.performer_nb_features,
        #         performer_feature_redraw_interval=agent_specification.performer_feature_redraw_interval,
        #         performer_generalized_attention=agent_specification.performer_generalized_attention,
        #         performer_auto_check_redraw=agent_specification.performer_auto_check_redraw,
        #     )
        else:
            print("unknown fe_type: ", agent_specification.fe_type)

        self.init_heads()

    @classmethod
    def load(cls, path, graphobs=True, max_n_modes=None):
        """Loading an agent corresponds to loading his model and a few args to specify how the model is working"""
        save_data = torch.load(path + "agent.pkl")
        agent_specification = save_data["agent_specification"]
        env_specification = save_data["env_specification"]
        if max_n_modes is not None:
            env_specification.max_n_modes = max_n_modes
            env_specification.max_n_nodes = max_n_modes
            env_specification.max_n_jobs = max_n_modes
        agent = cls(
            env_specification,
            agent_specification=agent_specification,
            graphobs=graphobs,
        )

        # constructors init weight!!!
        agent.gnn.load_state_dict(save_data["gnn"])
        agent.action_net.load_state_dict(save_data["action_net"])
        agent.value_net.load_state_dict(save_data["value_net"])
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
        # # action
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
        # usually ppo use gain = np.sqrt(2) here
        # best so far below
        # self.gnn.apply(
        #     partial(
        #         self.init_weights,
        #         gain=1.0,
        #         zero_bias=False,
        #         ortho_embed=self.agent_specification.ortho_embed,
        #     )
        # )
        # self.gnn.reset_egat()
        # usually ppo use gain = 0.01 here
        # self.action_net.apply(partial(self.init_weights, gain=1.0, zero_bias=False))
        # usually ppo use gain = 1 here
        # self.value_net.apply(partial(self.init_weights, gain=1.0, zero_bias=False))

        self.value_net.to(device)
        self.action_net.to(device)

    # def _obs_as_tensor_add_batch_dim(self, obs):
    #     return AgentObservation.np_to_torch(obs)

    # def _obs_as_tensor(self, obs):
    #     return AgentObservation.np_to_torch(obs)

    # def _rebatch_obs(self, obs):
    #     if isinstance(obs[0], str):
    #         return obs
    #     return AgentObservation.rebatch_obs(obs)

    def _obs_as_tensor_add_batch_dim_graph(self, obs):
        if self.env_specification.observe_subgraph:
            cobs = obs
        else:
            cobs = obs.clone()
        return AgentGraphObservation.rewire_internal(
            cobs,
            conflicts=self.agent_specification.conflicts,
            bidir=True,
            compute_laplacian_pe=False,
            laplacian_pe_cache=None,
            rwpe_k=self.agent_specification.rwpe_k,
            rwpe_cache=self.rwpe_cache,
        )

    def _obs_as_tensor_graph(self, obs):
        return [
            AgentGraphObservation.rewire_internal(
                o,
                conflicts=self.agent_specification.conflicts,
                bidir=True,
                compute_laplacian_pe=False,
                laplacian_pe_cache=None,
                rwpe_k=self.agent_specification.rwpe_k,
                rwpe_cache=self.rwpe_cache,
            )
            for o in obs
        ]

    def _rebatch_obs_graph(self, obs):
        # we need to flatten a list of list into a single list
        if isinstance(obs[0], str):
            return obs
        return sum(obs, [])

    def _get_obs_graph(self, b_obs, mb_ind):
        if isinstance(b_obs[0], str):
            # return [dgl.load_graphs(b_obs[i])[0][0] for i in mb_ind]
            # bobsi = [b_obs[i] for i in mb_ind]
            # return process_map(
            #     GraphFactory.load, bobsi, max_workers=16, chunksize=1, disable=True
            # )
            return [
                GraphFactory.load(b_obs[i], self.env_specification.pyg) for i in mb_ind
            ]
        return list(b_obs[i] for i in mb_ind)

    def _get_obs(self, b_obs, mb_ind):
        if isinstance(b_obs, list):
            list_obs = [pickle.load(open(b_obs[i], "rb")) for i in mb_ind]
            return self.rebatch_obs(list_obs)
        minibatched_obs = {}
        for key in b_obs:
            minibatched_obs[key] = b_obs[key][mb_ind]
        return minibatched_obs

    def preprocess(self, obs):
        # if self.graphobs:
        observation = AgentGraphObservation(
            list(obs) if isinstance(obs, tuple) else obs,
            conflicts=self.agent_specification.conflicts,
            factored_rp=self.env_specification.factored_rp,
            add_rp_edges=self.env_specification.add_rp_edges,
            max_n_resources=self.env_specification.max_n_resources,
            rwpe_k=self.agent_specification.rwpe_k,
            rwpe_cache=None,
            rewire_internal=False,
        )
        g = observation.graphs
        res_cal_id = observation.res_cal_id
        batch_size = observation.n_graphs
        num_nodes = observation.num_nodes
        if batch_size == 1:
            n_nodes = observation.batch_num_nodes[0]
        else:
            n_nodes = observation.batch_num_nodes
            origbnn = observation.batch_num_nodes
        # else:
        #     observation = AgentObservation(
        #         obs,
        #         conflicts=self.conflicts,
        #         add_self_loops=False,
        #         factored_rp=self.factored_rp,
        #         add_rp_edges=self.add_rp_edges,
        #         max_n_resources=self.max_n_resources,
        #         rwpe_k=self.rwpe_k,
        #         rwpe_cache=self.rwpe_cache,
        #     )
        #     batch_size = observation.get_batch_size()

        #     g = observation.to_graph()
        #     res_cal_id = observation.res_cal_id.flatten()

        #     n_nodes = observation.get_n_nodes()
        #     num_nodes = g.num_nodes()

        #     origbnn = g.batch_num_nodes()
        if (
            self.env_specification.remove_past_prec
            or self.env_specification.observe_subgraph
        ):
            if self.graphobs:
                g.add_self_loops()
                # else:
                #     g = dgl.add_self_loop(
                #         g,
                #         edge_feat_names=["type"],
                #         fill_data=AgentObservation.edgeType["self"],
                #     )
                # g.set_batch_num_nodes(origbnn)

        g, poolnodes, resource_nodes, vnodes = rewire(
            g,
            self.agent_specification.graph_pooling,
            self.agent_specification.conflicts == "node",
            self.agent_specification.vnode,
            batch_size,
            self.env_specification.n_features,
            self.env_specification.max_n_resources,
            6,
            7,
            10,
            10,
            8,
            9,
            self.graphobs,
        )

        if self.graphobs:
            g, felist = homogeneous_edges(
                g,
                AgentGraphObservation.edgeTypes,
                self.env_specification.factored_rp,
                self.agent_specification.conflicts,
                self.env_specification.max_n_resources,
            )
            g.to_homogeneous(ndata=["feat"], edata=felist)

        if self.agent_specification.rwpe_k != 0:
            edge_features_pe = self.edge_embedder_pe(g)
            g.ndata["pe"] = self.rwpe_embedder(
                torch.cat(
                    [
                        g.ndata["rwpe_global"],
                        g.ndata["rwpe_pr"],
                        g.ndata["rwpe_rp"],
                        g.ndata["rwpe_rc"],
                    ],
                    1,
                ),
            )
            pe = g.ndata["pe"]
        else:
            pe = None
        return (
            g,
            batch_size,
            num_nodes,
            n_nodes,
            poolnodes,
            resource_nodes,
            vnodes,
            res_cal_id,
            pe,
        )
