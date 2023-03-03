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
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .multihead_attention import MultiheadAttention
from .performer_pytorch import FastAttention


class MultiheadPerformerAttention(MultiheadAttention):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        performer_nb_features=None,
        performer_generalized_attention=False,
        attention_dropout=0.0,
        dropout=0.0,
        bias=True,
        self_attention=False,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            attention_dropout,
            dropout,
            bias,
            self_attention,
        )
        assert attention_dropout == 0.0
        self.fast_attention = FastAttention(
            self.head_dim,
            performer_nb_features,
            causal=False,
            generalized_attention=performer_generalized_attention,
            kernel_fn=nn.ReLU(),
            no_projection=False,
        )

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        return self.forward_performer(
            query, key, value, attn_bias, key_padding_mask, need_weights, attn_mask, before_softmax, need_head_weights
        )
