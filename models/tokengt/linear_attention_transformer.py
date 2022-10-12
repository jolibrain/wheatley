import torch
import torch.nn.functional as F
from torch import nn, einsum
import math
from operator import mul
from functools import partial

from .local_attention import LocalAttention

from einops import rearrange, repeat


# helper functions


def exists(val):
    return val is not None


def default(value, d):
    return d if not exists(value) else value


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


# self attention layer


def linear_attn(q, k, v, kv_mask=None):
    dim = q.shape[-1]

    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.0)
        del mask

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    q = q * dim**-0.5

    context = einsum("bhnd,bhne->bhde", k, v)
    attn = einsum("bhnd,bhde->bhne", q, context)
    return attn.reshape(*q.shape)


class LinearSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head=None,
        n_local_attn_heads=0,
        local_attn_window_size=128,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        assert dim_head or (dim % heads) == 0, "embedding dimension must be divisible by number of heads"
        d_heads = default(dim_head, dim // heads)

        self.heads = heads
        self.d_heads = d_heads

        self.global_attn_heads = heads - n_local_attn_heads
        self.global_attn_fn = linear_attn

        self.local_attn_heads = n_local_attn_heads
        self.local_attn = LocalAttention(local_attn_window_size, causal=False, dropout=attn_dropout)

        self.to_q = nn.Linear(dim, d_heads * heads, bias=False)

        kv_heads = heads

        self.kv_heads = kv_heads
        self.to_k = nn.Linear(dim, d_heads * kv_heads, bias=False)
        self.to_v = nn.Linear(dim, d_heads * kv_heads, bias=False)

        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_mask=None):

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        b, t, e, h, dh = *q.shape, self.heads, self.d_heads

        merge_heads = lambda x: x.reshape(*x.shape[:2], -1, dh).transpose(1, 2)

        q, k, v = map(merge_heads, (q, k, v))

        out = []

        split_index_fn = partial(split_at_index, 1, self.local_attn_heads)

        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))

        has_local, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

        if has_local:
            local_out = self.local_attn(lq, lk, lv, input_mask=~input_mask)
            out.append(local_out)

        if has_global:
            kv_mask = ~input_mask
            global_out = self.global_attn_fn(q, k, v, kv_mask=kv_mask)
            out.append(global_out)

        attn = torch.cat(out, dim=1)
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        return self.dropout(self.to_out(attn))
