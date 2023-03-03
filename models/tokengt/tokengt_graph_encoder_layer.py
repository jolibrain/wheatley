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

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import LayerNorm, Dropout

from .multihead_attention import MultiheadAttention
from .multihead_performer_attention import MultiheadPerformerAttention
from .linear_attention_transformer import LinearSelfAttention
from .feedforward import FeedForward
from .droppath import DropPath


class TokenGTGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        encoder_layers: int = 12,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        drop_path: float = 0.0,
        linear_transformer: bool = False,
        performer: bool = False,
        performer_nb_features: int = None,
        performer_generalized_attention: bool = False,
        activation_fn=torch.nn.functional.gelu,
        init_fn: Callable = None,
        layernorm_style: str = "postnorm",
        return_attention: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.layernorm_style = layernorm_style
        self.return_attention = return_attention

        self.linear_transformer = linear_transformer

        self.dropout_module = Dropout(dropout)

        # Initialize blocks
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            linear_transformer=linear_transformer,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            attention_dropout=attention_dropout,
            dropout=dropout,
            self_attention=True,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        # drop path for stochastic depth
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.feedforward = self.build_FFN(
            self.embedding_dim,
            ffn_embedding_dim,
            activation_fn,
            activation_dropout,
            dropout,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

        # drop path for stochastic depth
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def build_FFN(
        self,
        embedding_dim,
        ffn_embedding_dim,
        activation_fn,
        activation_dropout,
        dropout,
    ):
        return FeedForward(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            activation_fn=activation_fn,
            activation_dropout=activation_dropout,
            dropout=dropout,
        )

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        linear_transformer,
        performer,
        performer_nb_features,
        performer_generalized_attention,
        attention_dropout,
        dropout,
        self_attention,
    ):
        if linear_transformer:
            return LinearSelfAttention(
                embed_dim,
                num_attention_heads,
                n_local_attn_heads=0,
                local_attn_window_size=0,
                attn_dropout=attention_dropout,
                dropout=dropout,
            )
        elif performer:
            return MultiheadPerformerAttention(
                embed_dim,
                num_attention_heads,
                performer_nb_features=performer_nb_features,
                performer_generalized_attention=performer_generalized_attention,
                attention_dropout=attention_dropout,
                dropout=dropout,
                self_attention=True,
            )
        else:
            return MultiheadAttention(
                embed_dim,
                num_attention_heads,
                attention_dropout=attention_dropout,
                dropout=dropout,
                self_attention=True,
            )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        if self.layernorm_style == "prenorm":
            residual = x
            # x = self.self_attn_layer_norm(x)
            if self.linear_transformer:
                x = self.self_attn(x.transpose(0, 1), input_mask=self_attn_padding_mask)
                x = x.transpose(0, 1)
                attn = None
            else:
                x, attn = self.self_attn(
                    query=x,
                    key=x,
                    value=x,
                    attn_bias=self_attn_bias,
                    key_padding_mask=self_attn_padding_mask,
                    need_weights=self.return_attention,
                    need_head_weights=self.return_attention,
                    attn_mask=self_attn_mask,
                )
            x = self.dropout_module(x)
            x = self.drop_path1(x)
            x = residual + x

            residual = x
            # x = self.final_layer_norm(x)
            x = self.feedforward(x)
            x = self.drop_path2(x)
            x = residual + x

        elif self.layernorm_style == "postnorm":
            residual = x
            if self.linear_transformer:
                x = self.self_attn(x, input_mask=self_attn_padding_mask)
                attn = None
            else:
                x, attn = self.self_attn(
                    query=x,
                    key=x,
                    value=x,
                    attn_bias=self_attn_bias,
                    key_padding_mask=self_attn_padding_mask,
                    need_weights=self.return_attention,
                    need_head_weights=self.return_attention,
                    attn_mask=self_attn_mask,
                )
            x = self.dropout_module(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.feedforward(x)
            x = residual + x
            x = self.final_layer_norm(x)

        else:
            raise NotImplementedError
        return x, attn
