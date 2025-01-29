from typing import List, Literal, Tuple, Optional, Dict
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from collections import OrderedDict
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.models import MLP
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    spmm,
    scatter,
    remove_self_loops,
    add_remaining_self_loops,
)
from torch_geometric.nn.inits import glorot
from torch_geometric.utils.num_nodes import maybe_num_nodes

ACTIVATION_DICT = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
    "elu": nn.ELU(),
    "gelu": nn.GELU(),
    "none": None,
    "abs": torch.abs,
    "id": lambda x: x,
}


class HybridConv_v2(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_dim: int = None,
        channel_list: List[Tuple[int]] = [[0], [1], [2], [4], [0, 1], [1, 2], [2, 4]],
        combine_fn: Literal["att", "att_bias"] = "att_bias",
        num_heads: int = 1,
        activation_att1: str = "gelu",
        activation_att2: str = "gelu",
        activation: str = "gelu",
        depth_mlp: int = 1,
        skip: bool = True,
        add_self_loops: bool = True,
        norm: str = "gcn",
        filter_norm_dim: bool = None,
        bias: bool = True,
    ):
        super().__init__(aggr="add")
        self.aggr = "add"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_list = channel_list
        self.channel_low = [channel for channel in channel_list if len(channel) == 1]
        self.channel_band = [channel for channel in channel_list if len(channel) == 2]
        self.radius_list = list(
            set([agg for channel in channel_list for agg in channel])
        )
        self.radius_list.sort()
        self.combine_fn = combine_fn
        self.num_heads = num_heads
        self.skip = skip
        self.add_self_loops = True
        self.norm = norm
        self.filter_norm_dim = filter_norm_dim
        self.activation_att1 = ACTIVATION_DICT[activation_att1]
        self.activation_att2 = ACTIVATION_DICT[activation_att2]
        self.activation = ACTIVATION_DICT[activation]

        self.edge_dim = edge_dim if edge_dim is not None else input_dim
        self.to_weight = MLP(
            [self.edge_dim] * 2 + [1], bias=bias, activation=activation, norm=None
        )

        self.head_dict = nn.ModuleDict()
        for channel in channel_list:
            self.head_dict[str(channel)] = nn.ModuleList()
            for _ in range(num_heads):
                self.head_dict[str(channel)].append(
                    nn.Linear(input_dim, output_dim, bias=bias)
                )
        if len(self.channel_low) > 0:
            self.att_pre_low = nn.Parameter(torch.empty(num_heads, output_dim))
            self.att_channel_low = nn.Parameter(torch.empty(num_heads, output_dim))
        if len(self.channel_band) > 0:
            self.att_pre_band = nn.Parameter(torch.empty(num_heads, output_dim))
            self.att_channel_band = nn.Parameter(torch.empty(num_heads, output_dim))
        if depth_mlp > 0:
            m = 0
            if len(self.channel_low) > 0:
                m += 1
            if len(self.channel_band) > 0:
                m += 1

            if combine_fn == "att_bias":
                m = 1

            if skip:
                m += 1

            self.mlp_out = MLP(
                [m * output_dim] + depth_mlp * [output_dim],
                bias=bias,
                activation=activation,
                norm=None,
            )
        else:
            self.mlp_out = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for channel in self.channel_list:
            for k in range(self.num_heads):
                self.head_dict[str(channel)][k].reset_parameters()
        if len(self.channel_low) > 0:
            glorot(self.att_pre_low)
            glorot(self.att_channel_low)
        if len(self.channel_band) > 0:
            glorot(self.att_pre_band)
            glorot(self.att_channel_band)
        if self.mlp_out is not None:
            self.mlp_out.reset_parameters()

    def forward(self, x, edge_index, edge_feats) -> Tensor:
        if edge_feats is not None:
            edge_weight = torch.sigmoid(self.to_weight(edge_feats).squeeze(-1))
        else:
            edge_weight = None

        if self.norm == "gcn":
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index,
                edge_weight,
                x.size(self.node_dim),
                False,
                self.add_self_loops,
                self.flow,
                dtype=x.dtype,
            )
        elif self.norm == "rw":
            edge_index, edge_weight = rw_norm(  # yapf: disable
                edge_index,
                edge_weight,
                x.size(self.node_dim),
                False,
                self.add_self_loops,
                self.flow,
                dtype=x.dtype,
            )
        elif self.norm == "sym":
            edge_index, edge_weight = sym_norm(  # yapf: disable
                edge_index,
                edge_weight,
                x.size(self.node_dim),
                False,
                self.add_self_loops,
                self.flow,
                dtype=x.dtype,
            )
        elif self.norm == "avg":
            edge_index, edge_weight = avg_norm(  # yapf: disable
                edge_index,
                edge_weight,
                x.size(self.node_dim),
                False,
                self.add_self_loops,
                self.flow,
                dtype=x.dtype,
            )
        else:
            raise NotImplementedError("norm type not supported")

        r_tmp = 0
        x_agg_dict = OrderedDict()
        x_agg_dict[0] = x
        for this_r in self.radius_list:
            if this_r == 0:
                continue
            x = list(x_agg_dict.values())[-1]
            for _ in range(this_r - r_tmp):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            x_agg_dict[this_r] = x
            r_tmp = this_r

        x_channel_dict = {}

        for channel in self.channel_list:
            if len(channel) == 1:
                x_channel_dict[str(channel)] = self.normalize_filter(
                    x_agg_dict[channel[0]], dim=self.filter_norm_dim
                )
            else:
                x_channel_dict[str(channel)] = self.normalize_filter(
                    x_agg_dict[channel[0]] - x_agg_dict[channel[1]],
                    dim=self.filter_norm_dim,
                )

        if self.combine_fn == "att":
            x = torch.cat(self.channel_attention(x_channel_dict), dim=-1)

        elif self.combine_fn == "att_bias":
            bias = torch.stack(self.channel_attention(x_channel_dict), dim=-1).sum(
                dim=-1
            )
            x = x + bias

        else:
            raise ValueError("combine_fn not supported")

        if self.skip:
            x = torch.cat([x_agg_dict[0], x], dim=-1)

        if self.mlp_out is not None:
            x = self.mlp_out(x)
        else:
            x = self.activation(x)

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def channel_attention(self, x_channel_dict: Dict) -> Tensor:
        x_channel_low, x_channel_band = [], []
        for channel, h in x_channel_dict.items():
            channel = eval(channel)
            if len(channel) == 1:
                heads = []
                for k in range(self.num_heads):
                    heads.append(
                        self.activation_att1(self.head_dict[str(channel)][k](h))
                    )
                x_channel_low.append(torch.stack(heads, dim=0))
            else:
                heads = []
                for k in range(self.num_heads):
                    heads.append(
                        self.activation_att1(self.head_dict[str(channel)][k](h))
                    )
                x_channel_band.append(torch.stack(heads, dim=0))

        x_low, x_band = None, None
        if len(x_channel_low) > 0:
            e_pre_low = self.activation_att2(
                torch.matmul(x_channel_low[0], self.att_pre_low.unsqueeze(-1))
            )
            e_channel_list_low = [
                self.activation_att2(
                    torch.matmul(x_channel, self.att_channel_low.unsqueeze(-1))
                )
                for x_channel in x_channel_low[1:]
            ]
            e_low = torch.stack(e_channel_list_low, dim=0) + e_pre_low
            channel_weights_low = torch.softmax(e_low, dim=0)
            weighted_channels_low = torch.mul(
                channel_weights_low, torch.stack(x_channel_low[1:], dim=0)
            )
            x_low = weighted_channels_low.sum(dim=0).mean(dim=0)
        if len(x_channel_band) > 0:
            e_pre_band = self.activation_att2(
                torch.matmul(x_channel_band[0], self.att_pre_band.unsqueeze(-1))
            )
            e_channel_list_band = [
                self.activation_att2(
                    torch.matmul(x_channel, self.att_channel_band.unsqueeze(-1))
                )
                for x_channel in x_channel_band
            ]
            e_band = torch.stack(e_channel_list_band, dim=0) + e_pre_band
            channel_weights_band = torch.softmax(e_band, dim=0)
            weighted_channels_band = torch.mul(
                channel_weights_band, torch.stack(x_channel_band, dim=0)
            )
            x_band = weighted_channels_band.sum(dim=0).mean(dim=0)

        combine = []
        if x_low is not None:
            combine.append(x_low)
        if x_band is not None:
            combine.append(x_band)

        return combine

        # x = torch.cat(combine, dim=-1)

        # return x

    def normalize_filter(self, x: Tensor, dim: int = 1, eps: float = 1e-5) -> Tensor:
        if dim in [0, 1]:
            mean, var = x.mean(dim), x.var(dim)
            x = (x - mean.unsqueeze(dim)) / torch.sqrt(var + eps).unsqueeze(dim)

        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.input_dim}, "
            f"{self.output_dim}, channel_list={self.channel_list})"
        )


class HybridConvLayer(nn.Module):
    """HybridConv layer"""

    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.model = HybridConv_v2(
            in_dim,
            out_dim,
            num_heads=num_heads,
        )

    def forward(self, g, node_feats, edge_feats):
        return self.model(node_feats, g.edge_index, edge_feats)


def rw_norm(
    edge_index,
    edge_weight=None,
    num_nodes=None,
    improved=False,
    add_self_loops=True,
    flow="source_to_target",
    dtype=None,
):
    fill_value = 2.0 if improved else 1.0

    assert flow in ["source_to_target", "target_to_source"]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_index_wo_self_loops, _ = remove_self_loops(edge_index)

    if add_self_loops:
        edge_index_w_self_loops, edge_weight = add_remaining_self_loops(
            edge_index_wo_self_loops, edge_weight, fill_value, num_nodes
        )

    if edge_weight is None:
        edge_weight_w_self_loops = torch.ones(
            (edge_index_w_self_loops.size(1),),
            dtype=dtype,
            device=edge_index_w_self_loops.device,
        )
        edge_weight_wo_self_loops = torch.ones(
            (edge_index_wo_self_loops.size(1),),
            dtype=dtype,
            device=edge_index_wo_self_loops.device,
        )

    deg = scatter(
        edge_weight_wo_self_loops,
        edge_index_wo_self_loops[1],
        dim=0,
        dim_size=num_nodes,
        reduce="sum",
    )
    deg_inv = deg.pow_(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)
    edge_weight = edge_weight_w_self_loops * deg_inv[edge_index_w_self_loops[1]]

    self_loop_idx = edge_index_w_self_loops[0] == edge_index_w_self_loops[1]
    edge_weight[self_loop_idx] = 1.0

    return edge_index_w_self_loops, edge_weight / 2


def sym_norm(
    edge_index,
    edge_weight=None,
    num_nodes=None,
    improved=False,
    add_self_loops=True,
    flow="source_to_target",
    dtype=None,
):
    fill_value = 2.0 if improved else 1.0

    assert flow in ["source_to_target", "target_to_source"]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_index_wo_self_loops, _ = remove_self_loops(edge_index)

    if add_self_loops:
        edge_index_w_self_loops, edge_weight = add_remaining_self_loops(
            edge_index_wo_self_loops, edge_weight, fill_value, num_nodes
        )

    if edge_weight is None:
        edge_weight_w_self_loops = torch.ones(
            (edge_index_w_self_loops.size(1),),
            dtype=dtype,
            device=edge_index_w_self_loops.device,
        )
        edge_weight_wo_self_loops = torch.ones(
            (edge_index_wo_self_loops.size(1),),
            dtype=dtype,
            device=edge_index_wo_self_loops.device,
        )

    deg = scatter(
        edge_weight_wo_self_loops,
        edge_index_wo_self_loops[1],
        dim=0,
        dim_size=num_nodes,
        reduce="sum",
    )
    deg_inv_sqr = deg.pow_(-0.5)
    deg_inv_sqr.masked_fill_(deg_inv_sqr == float("inf"), 0)
    edge_weight = (
        deg_inv_sqr[edge_index_w_self_loops[0]]
        * edge_weight_w_self_loops
        * deg_inv_sqr[edge_index_w_self_loops[1]]
    )

    self_loop_idx = edge_index_w_self_loops[0] == edge_index_w_self_loops[1]
    edge_weight[self_loop_idx] = 1.0

    return edge_index_w_self_loops, edge_weight / 2


def avg_norm(
    edge_index,
    edge_weight=None,
    num_nodes=None,
    improved=False,
    add_self_loops=True,
    flow="source_to_target",
    dtype=None,
):
    fill_value = 2.0 if improved else 1.0

    assert flow in ["source_to_target", "target_to_source"]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_index_wo_self_loops, _ = remove_self_loops(edge_index)

    if add_self_loops:
        edge_index_w_self_loops, edge_weight = add_remaining_self_loops(
            edge_index_wo_self_loops, edge_weight, fill_value, num_nodes
        )

    if edge_weight is None:
        edge_weight_w_self_loops = torch.ones(
            (edge_index_w_self_loops.size(1),),
            dtype=dtype,
            device=edge_index_w_self_loops.device,
        )
        edge_weight_wo_self_loops = torch.ones(
            (edge_index_wo_self_loops.size(1),),
            dtype=dtype,
            device=edge_index_wo_self_loops.device,
        )

    deg = scatter(
        edge_weight_wo_self_loops,
        edge_index_wo_self_loops[1],
        dim=0,
        dim_size=num_nodes,
        reduce="sum",
    )
    deg_inv = deg.pow_(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)
    edge_weight = edge_weight_w_self_loops * deg_inv[edge_index_w_self_loops[0]]

    self_loop_idx = edge_index_w_self_loops[0] == edge_index_w_self_loops[1]
    edge_weight[self_loop_idx] = 1.0

    return edge_index_w_self_loops, edge_weight / 2
