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

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .tokengt.tokengt_graph_encoder import init_graphormer_params, TokenGTGraphEncoder
import torch
from torch.nn import LayerNorm
from utils.agent_observation import AgentObservation


class FeaturesExtractorTokenGT(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        input_dim_features_extractor,
        device,
        max_n_nodes,
        max_n_machines,
        conflicts="att",
        orf_node_id=False,
        orf_node_id_dim=64,
        lap_node_id=True,
        lap_node_id_k=16,
        lap_node_id_sign_flip=False,
        lap_node_id_eig_dropout=0.0,
        cache_lap_node_id=False,
        type_id=True,
        stochastic_depth=False,
        transformer_flavor="linear",
        performer_nb_features=None,
        performer_feature_redraw_interval=100,
        performer_generalized_attention=False,
        performer_auto_check_redraw=True,
        encoder_layers=12,
        encoder_embed_dim=768,
        encoder_ffn_embed_dim=768,
        encoder_attention_heads=32,
        dropout=0.0,
        attention_dropout=0.0,
        act_dropout=0.0,
        encoder_normalize_before=False,
        layernorm_style="prenorm",
        apply_graphormer_init=True,
        activation_fn=torch.nn.functional.gelu,
        layer_pooling="last",
    ):

        self.layer_pooling = layer_pooling
        linear_transformer = False
        if cache_lap_node_id:
            self.laplacian_pe_cache = {}
        else:
            self.laplacian_pe_cache = None

        self.lap_node_id_k = lap_node_id_k
        performer = False
        if transformer_flavor == "linear":
            linear_transformer = True
        elif transformer_flavor == "performer":
            performer = True
        if self.layer_pooling == "all":
            features_dim = encoder_embed_dim * (encoder_layers + 1) * 2
        else:
            features_dim = encoder_embed_dim * 2
        super(FeaturesExtractorTokenGT, self).__init__(
            observation_space=observation_space,
            features_dim=features_dim,
        )
        self.max_n_machines = max_n_machines
        self.max_n_nodes = max_n_nodes
        self.conflicts = conflicts
        self.device = device
        self.laplacian_pe = lap_node_id
        self.graph_encoder = TokenGTGraphEncoder(
            # <
            input_dim_features_extractor=input_dim_features_extractor,  # to be embedded
            num_edges_type=5 + max_n_machines + 1,  # num (edge type) !
            # >
            # < for tokenization
            orf_node_id=orf_node_id,
            orf_node_id_dim=orf_node_id_dim,
            lap_node_id=lap_node_id,
            lap_node_id_k=lap_node_id_k,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            type_id=type_id,
            # >
            # <
            stochastic_depth=stochastic_depth,
            linear_transformer=linear_transformer,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_feature_redraw_interval=performer_feature_redraw_interval,
            performer_generalized_attention=performer_generalized_attention,
            performer_auto_check_redraw=performer_auto_check_redraw,
            num_encoder_layers=encoder_layers,
            embedding_dim=encoder_embed_dim,
            ffn_embedding_dim=encoder_ffn_embed_dim,
            num_attention_heads=encoder_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=act_dropout,
            encoder_normalize_before=encoder_normalize_before,
            layernorm_style=layernorm_style,
            apply_graphormer_init=apply_graphormer_init,
            activation_fn=activation_fn,
            # >
        )

        self.lm_head_transform_weight = torch.nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.activation_fn = activation_fn
        self.layer_norm = LayerNorm(encoder_embed_dim)

    def forward(self, obs):

        # print("lap cache size", len(self.laplacian_pe_cache))
        g_list = AgentObservation.from_gym_observation(
            obs,
            conflicts=self.conflicts,
            max_n_machines=self.max_n_machines,
            add_self_loops=False,
            device=self.device,
            do_batch=False,
            compute_laplacian_pe=self.laplacian_pe,
            laplacian_pe_cache=self.laplacian_pe_cache,
            n_laplacian_eigv=self.lap_node_id_k,
            bidir=False,
        ).to_graph()

        max_node_num = max([gr.num_nodes() for gr in g_list])

        inner_states, graph_rep, attn_dict = self.graph_encoder(g_list, last_state_only=(self.layer_pooling == "last"))

        if self.layer_pooling == "last":
            x = inner_states[-1]  # B x T x C
            # last LN does no good
            # x = self.layer_norm(self.activation_fn()(self.lm_head_transform_weight(x)))
            x = self.lm_head_transform_weight(x)
            node_rep = x[:, 2 : max_node_num + 2, :]
        else:
            graph_reps = [rep[:, 0, :] for rep in inner_states]
            graph_rep = torch.cat(graph_reps, dim=1)
            # istates = [self.layer_norm(self.activation_fn()(self.lm_head_transform_weight(rep))) for rep in inner_states]
            # last LN does no good
            istates = [self.activation_fn()(self.lm_head_transform_weight(rep)) for rep in inner_states]
            node_rep = torch.cat(istates, dim=2)[:, 2 : max_node_num + 2, :]
        node_rep = torch.nn.functional.pad(
            node_rep, (0, 0, 0, self.max_n_nodes - node_rep.shape[1]), mode="constant", value=0.0
        )
        graph_rep = graph_rep.unsqueeze(1).expand(-1, node_rep.shape[1], -1)
        ret = torch.cat([node_rep, graph_rep], dim=2)
        # ret  =torch.nn.functional.pad(ret, (0, 0, 0, self.max_n_nodes - ret.shape[1]), mode="constant", value=0.0)
        return ret
