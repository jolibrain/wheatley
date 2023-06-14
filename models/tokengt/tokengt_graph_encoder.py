#    MIT License
#
#    Copyright (c) Microsoft Corporation.
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE
#
"""
Modified from https://github.com/microsoft/Graphormer
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import LayerNorm, Dropout

from .performer_pytorch import ProjectionUpdater
from .multihead_attention import MultiheadAttention
from .linear_attention_transformer import LinearSelfAttention
from .tokenizer import GraphFeatureTokenizer
from .tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer


def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)
    if isinstance(module, LinearSelfAttention):
        normal_(module.to_q.weight.data)
        normal_(module.to_k.weight.data)
        normal_(module.to_v.weight.data)
        # normal_(module.to_out.weight.data)


class TokenGTGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim_features_extractor: int,
        num_edges_type: int = -1,
        max_n_resources: int = -1,
        orf_node_id: bool = False,
        orf_node_id_dim: int = 64,
        lap_node_id: bool = False,
        lap_node_id_k: int = 8,
        lap_node_id_sign_flip: bool = False,
        lap_node_id_eig_dropout: float = 0.0,
        type_id: bool = False,
        stochastic_depth: bool = False,
        linear_transformer=False,
        performer: bool = False,
        performer_nb_features: int = None,
        performer_feature_redraw_interval: int = 1000,
        performer_generalized_attention: bool = False,
        performer_auto_check_redraw: bool = True,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        encoder_normalize_before: bool = False,
        layernorm_style: str = "prenorm",
        apply_graphormer_init: bool = False,
        activation_fn=torch.nn.functional.gelu,
        embed_scale: float = None,
        return_attention: bool = False,
        permute=False,
        factored_rp=False,
    ) -> None:

        super().__init__()
        self.dropout_module = Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.performer = performer
        self.linear_transformer = linear_transformer

        self.permute = permute

        self.graph_feature = GraphFeatureTokenizer(
            input_dim_features_extractor=input_dim_features_extractor,
            num_edges_type=num_edges_type,
            max_n_resources=max_n_resources,
            orf_node_id=orf_node_id,
            orf_node_id_dim=orf_node_id_dim,
            lap_node_id=lap_node_id,
            lap_node_id_k=lap_node_id_k,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            type_id=type_id,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            factored_rp=factored_rp,
        )
        self.embed_scale = embed_scale

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        self.layers = nn.ModuleList([])

        if stochastic_depth:
            assert layernorm_style == "prenorm"  # only for residual nets

        self.layers.extend(
            [
                self.build_tokengt_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    encoder_layers=num_encoder_layers,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    drop_path=(0.1 * (layer_idx + 1) / num_encoder_layers)
                    if stochastic_depth
                    else 0,
                    linear_transformer=linear_transformer,
                    performer=performer,
                    performer_nb_features=performer_nb_features,
                    performer_generalized_attention=performer_generalized_attention,
                    activation_fn=activation_fn,
                    layernorm_style=layernorm_style,
                    return_attention=return_attention,
                )
                for layer_idx in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        if performer:
            # keeping track of when to redraw projections for all attention layers
            self.performer_auto_check_redraw = performer_auto_check_redraw
            self.performer_proj_updater = ProjectionUpdater(
                self.layers, performer_feature_redraw_interval
            )

    def build_tokengt_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        encoder_layers,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        drop_path,
        linear_transformer,
        performer,
        performer_nb_features,
        performer_generalized_attention,
        activation_fn,
        layernorm_style,
        return_attention,
    ):
        return TokenGTGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            encoder_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            drop_path=drop_path,
            linear_transformer=linear_transformer,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            activation_fn=activation_fn,
            layernorm_style=layernorm_style,
            return_attention=return_attention,
        )

    def forward(
        self,
        listed_data,
        perturb=None,
        last_state_only: bool = False,
    ):

        if self.performer and self.performer_auto_check_redraw:
            self.performer_proj_updater.redraw_projections()

        x, padding_mask, padded_index = self.graph_feature(listed_data, perturb)
        # print("xshape", x.shape)
        # print("padding_mask.shape", padding_mask.shape)
        # print("padded_index.shape", padded_index.shape)
        if self.permute:
            perm = torch.empty((x.shape[0], x.shape[1] - 2), dtype=torch.long)
            invperm = torch.empty((x.shape[0], x.shape[1] - 2), dtype=torch.long)
            for i in range(perm.shape[0]):
                perm[i] = torch.randperm(x.shape[1] - 2)
                invperm[i] = torch.argsort(perm[i])
            newx = torch.empty_like(x)
            newx[:, :2, :] = x[:, :2, :]
            for i in range(x.shape[0]):
                newx[i, 2:, :] = x[i, perm[i] + 2, :]
            x = newx

        # x: B x T x C

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        attn_dict = {"maps": {}, "padded_index": padded_index}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=None,
                self_attn_bias=None,
            )
            if not last_state_only:
                y = x.transpose(0, 1)
                if self.permute:
                    newy = torch.empty_like(y)
                    newy[:, :2, :] = y[:, :2, :]
                    for i in range(y.shape[0]):
                        newy[i, 2:, :] = y[i, invperm[i] + 2, :]
                    y = newy
                inner_states.append(y)
            attn_dict["maps"][i] = attn

        graph_rep = x[0, :, :]

        if last_state_only:
            y = x.transpose(0, 1)
            if self.permute:
                newy = torch.empty_like(y)
                newy[:, :2, :] = y[:, :2, :]
                for i in range(y.shape[0]):
                    newy[i, :2] = y[i, invperm[i] + 2]
                y = newy
            inner_states = [y]

        return inner_states, graph_rep, attn_dict
