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


class AgentSpecification:
    def __init__(
        self,
        n_features,
        gconv_type,
        graph_has_relu,
        graph_pooling,
        layer_pooling,
        mlp_act,
        mlp_act_graph,
        device,
        n_mlp_layers_features_extractor,
        n_layers_features_extractor,
        hidden_dim_features_extractor,
        n_attention_heads,
        reverse_adj,
        residual_gnn,
        normalize_gnn,
        conflicts,
        n_mlp_layers_actor,
        hidden_dim_actor,
        n_mlp_layers_critic,
        hidden_dim_critic,
        fe_type,
        transformer_flavor,
        dropout,
        cache_lap_node_id,
        lap_node_id_k,
        edge_embedding_flavor,
        performer_nb_features,
        performer_feature_redraw_interval,
        performer_generalized_attention,
        performer_auto_check_redraw,
        vnode,
        update_edge_features,
        update_edge_features_pe,
        ortho_embed,
        no_tct,
        mid_in_edges,
        rwpe_k,
        rwpe_h,
        cache_rwpe,
        two_hot,
        symlog,
        hl_gauss,
        reward_weights,
        sgformer,
        pyg,
        hierarchical,
        tokengt=False,
        shared_conv=False,
        checkpoint=1,
        dual_net=False,
    ):
        self.n_features = n_features
        self.gconv_type = gconv_type
        self.graph_has_relu = graph_has_relu
        self.graph_pooling = graph_pooling
        self.layer_pooling = layer_pooling
        self.mlp_act = mlp_act
        self.mlp_act_graph = mlp_act_graph
        self.device = device
        self.n_mlp_layers_features_extractor = n_mlp_layers_features_extractor
        self.n_layers_features_extractor = n_layers_features_extractor
        self.hidden_dim_features_extractor = hidden_dim_features_extractor
        self.n_attention_heads = n_attention_heads
        self.reverse_adj = reverse_adj
        self.residual_gnn = residual_gnn
        self.normalize_gnn = normalize_gnn
        self.conflicts = conflicts
        self.n_mlp_layers_actor = n_mlp_layers_actor
        self.hidden_dim_actor = hidden_dim_actor
        self.n_mlp_layers_critic = n_mlp_layers_critic
        self.hidden_dim_critic = hidden_dim_critic
        self.fe_type = fe_type
        self.transformer_flavor = transformer_flavor
        self.dropout = dropout
        self.cache_lap_node_id = cache_lap_node_id
        self.lap_node_id_k = lap_node_id_k
        self.edge_embedding_flavor = edge_embedding_flavor
        self.performer_nb_features = performer_nb_features
        self.performer_feature_redraw_interval = performer_feature_redraw_interval
        self.performer_generalized_attention = performer_generalized_attention
        self.performer_auto_check_redraw = performer_auto_check_redraw
        self.vnode = vnode
        self.update_edge_features = update_edge_features
        self.update_edge_features_pe = update_edge_features_pe
        self.ortho_embed = ortho_embed
        self.no_tct = no_tct
        self.mid_in_edges = mid_in_edges
        self.rwpe_k = rwpe_k
        self.rwpe_h = rwpe_h
        self.cache_rwpe = cache_rwpe
        self.two_hot = two_hot
        self.symlog = symlog
        self.hl_gauss = hl_gauss
        self.reward_weights = reward_weights
        self.sgformer = sgformer
        self.pyg = pyg
        self.hierarchical = hierarchical
        self.tokengt = tokengt
        self.shared_conv = shared_conv
        self.checkpoint = checkpoint
        self.dual_net = dual_net

        if mlp_act.lower() == "relu":
            self.activation_fn = torch.nn.LeakyReLU
        elif mlp_act.lower() == "tanh":
            self.activation_fn = torch.nn.Tanh
        elif mlp_act.lower() == "elu":
            self.activation_fn = torch.nn.ELU
        elif mlp_act.lower() == "gelu":
            self.activation_fn = torch.nn.GELU
        elif mlp_act.lower() == "selu":
            self.activation_fn = torch.nn.SELU
        else:
            raise Exception("Activation not recognized")

        self.mlp_act_graph = mlp_act_graph
        if mlp_act_graph.lower() == "relu":
            self.activation_fn_graph = torch.nn.LeakyReLU
        elif mlp_act_graph.lower() == "tanh":
            self.activation_fn_graph = torch.nn.Tanh
        elif mlp_act_graph.lower() == "elu":
            self.activation_fn_graph = torch.nn.ELU
        elif mlp_act_graph.lower() == "gelu":
            self.activation_fn_graph = torch.nn.GELU
        elif mlp_act_graph.lower() == "selu":
            self.activation_fn_graph = torch.nn.SELU
        elif mlp_act_graph.lower() == "silu":
            self.activation_fn_graph = torch.nn.SiLU
        else:
            raise Exception("Activation not recognized")

        pi = [hidden_dim_actor] * n_mlp_layers_actor
        vf = [hidden_dim_critic] * n_mlp_layers_critic
        self.net_arch = dict(vf=vf, pi=pi)

    def print_self(self):
        print(
            f"==========Agent Description   ==========\n"
            f"Features extractor type:          {self.fe_type}\n"
            f"Layer Pooling:                    {self.layer_pooling}\n"
            f"Dropout:                          {self.dropout}\n"
            f"Ortho embed:                      {self.ortho_embed}\n"
            f"Remove tct:                       {self.no_tct}\n"
        )
        if self.fe_type == "tokengt":
            print(f"Net shapes:")
            shape = f"{self.n_features} -> ( {self.hidden_dim_features_extractor} / {self.n_attention_heads} ) x {self.n_layers_features_extractor}"

            actor_shape = (
                f""
                + "".join(
                    [
                        f" -> {self.hidden_dim_actor}"
                        for _ in range(self.n_mlp_layers_actor)
                    ]
                )
                + " -> 1"
            )
            critic_shape = (
                f""
                + "".join(
                    [
                        f" -> {self.hidden_dim_critic}"
                        for _ in range(self.n_mlp_layers_critic)
                    ]
                )
                + " -> 1"
            )
            print(f" - Features extractor: TokenGT/{self.mlp_act_graph} {shape}")
            print(f" - Actor: {actor_shape}")
            print(f" - Critic: {critic_shape}\n")
        else:
            print(
                f"Convolutional FE\n"
                f"Graph convolution type:           {self.gconv_type.upper()}\n"
                f"Use pyg:                          {self.pyg}\n"
                f"Add (R)eLU between graph layers:  {'Yes' if self.graph_has_relu else 'No'}\n"
                f"Graph pooling type:               {self.graph_pooling.title()}\n"
                f"Activation function of agent:     {self.mlp_act.title()}\n"
                f"Activation function of graph:     {self.mlp_act_graph.title()}\n"
                f"Use VNode:                        {self.vnode}\n"
                f"Machine id in edges:              {self.mid_in_edges}\n"
                f"Update edge features:             {self.update_edge_features}\n"
                f"Update edge features (PE):        {self.update_edge_features_pe}\n"
                f"RWPE k:                           {self.rwpe_k}\n"
                f"RWPE h:                           {self.rwpe_h}\n"
                f"RWPE cache:                       {self.cache_rwpe}\n"
                f"two hot encoding:                 {self.two_hot}\n"
                f"symlog critic:                    {self.symlog}\n"
                f"hl gauss encoding:                {self.hl_gauss}\n"
                f"reward weights:                   {self.reward_weights}\n"
                f"SGFormer:                         {self.sgformer}\n"
                f"Hierarchical:                     {self.hierarchical}\n"
                f"TokenGT:                          {self.tokengt}\n"
                f"Shared conv:                      {self.shared_conv}\n"
                f"checkpoint:                       {self.checkpoint}\n"
                f"Dual Net:                         {self.dual_net}\n"
                f"Net shapes:"
            )
            first_features_extractor_shape = (
                f"{self.n_features} -> {self.hidden_dim_features_extractor}"
            )

            other_features_extractor_shape = f"{self.hidden_dim_features_extractor} -> {self.hidden_dim_features_extractor}"

            actor_shape = (
                f""
                + "".join(
                    [
                        f" -> {self.hidden_dim_actor}"
                        for _ in range(self.n_mlp_layers_actor)
                    ]
                )
                + " -> 1"
            )
            critic_shape = (
                f""
                + "".join(
                    [
                        f" -> {self.hidden_dim_critic}"
                        for _ in range(self.n_mlp_layers_critic)
                    ]
                )
                + " -> 1"
            )
            print(
                f" - Features extractor: {self.gconv_type.upper()}({first_features_extractor_shape}) => "
                + f"{self.gconv_type.upper()}({other_features_extractor_shape}) x {self.n_layers_features_extractor - 1}"
            )
            print(f" - Actor: {actor_shape}")
            print(f" - Critic: {critic_shape}\n")
