import torch
from torch import nn
import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import function as fn
import time


def to_dense_adj(edge_index, edge_attr=None, num_nodes=None, total_edge_index=None):
    if edge_attr is None:
        edge_attr = torch.ones(edge_index[0].size(0))
    if num_nodes is None:
        num_nodes = torch.stack(edge_index, dim=0).max().item() + 1

    if len(edge_attr.size()) == 1:
        dense = torch.zeros([num_nodes, num_nodes], dtype=edge_attr.dtype)
    elif len(edge_attr.size()) == 2:
        dense = torch.zeros(
            [num_nodes, num_nodes, edge_attr.size(1)], dtype=edge_attr.dtype
        )
    else:
        raise ValueError("The shape of edge_attr is invalid.")

    dense[edge_index] = edge_attr
    if total_edge_index is not None:
        dense = dense[total_edge_index]
    return dense


def power_computation(power, graph_matrix):
    if isinstance(power, list):
        left = power[0]
        right = power[1]
    else:
        left = 1
        right = power
    if left <= 0 or right <= 0 or left > right:
        raise ValueError("Invalid power {}".format(power))

    bases = []
    graph_matrix_n = torch.eye(graph_matrix.shape[0], dtype=graph_matrix.dtype)
    for _ in range(left - 1):
        graph_matrix_n = torch.matmul(graph_matrix_n, graph_matrix)
    for _ in range(left, right + 1):
        graph_matrix_n = torch.matmul(graph_matrix_n, graph_matrix)
        bases = bases + [graph_matrix_n]
    return bases


def basis_transform(
    g, basis, power, epsilon, degs, edgehop=None, basis_norm=False, basis_cache=None
):
    # check_edge_max_equal_num_nodes(g)  # with self_loop added, this should be held
    # check_repeated_edges(g)
    if basis_cache is not None:
        edges = g.edges(order="srcdst")
        key = (
            g.num_nodes(),
            *(edges[0].tolist() + edges[1].tolist()),
        )
        if key in basis_cache:
            return basis_cache[key]

    edge_idx = g.edges()
    adj = to_dense_adj(edge_idx)  # Graphs may have only one node.
    # check_dense(g, adj)

    bases = [torch.eye(adj.shape[0], dtype=adj.dtype)]
    deg = adj.sum(1)
    for i, eps in enumerate(epsilon):
        sym_basis = deg.pow(eps).unsqueeze(-1)
        graph_matrix = torch.matmul(sym_basis, sym_basis.transpose(0, 1)) * adj
        bases = bases + power_computation(power[i], graph_matrix)
    for e in degs:
        bases = bases + [deg.pow(e).diag()]

    if basis == "DEN":
        for i in range(len(bases)):
            bases[i] = bases[i].flatten(0)
        new_edge_idx = torch.ones_like(adj, dtype=adj.dtype).nonzero(as_tuple=True)
    elif basis == "SPA":
        if edgehop is None:
            edgehop = max(
                [
                    power[i][-1] if isinstance(power[i], list) else power[i]
                    for i in range(len(power))
                ]
            )
        # print('Max power {}'.format(max_power))
        pos_edge_idx = adj
        for i in range(1, edgehop):
            pos_edge_idx = torch.matmul(pos_edge_idx, adj)
        pos_edge_idx = pos_edge_idx.nonzero(as_tuple=True)
        for i in range(len(bases)):
            bases[i] = bases[i][pos_edge_idx]
        new_edge_idx = pos_edge_idx
    else:
        raise ValueError("Unknown basis called {}".format(basis))

    bases = torch.stack(bases, dim=0)
    if basis_norm:
        std = torch.std(bases, 1, keepdim=True, unbiased=False)
        mean = torch.mean(bases, 1, keepdim=True)
        bases = (bases - mean) / (std + 1e-5)
        # bases = bases - mean
    bases = bases.transpose(-2, -1).contiguous()

    # print(new_edge_idx)
    new_g = dgl.graph(new_edge_idx)
    assert new_g.num_nodes() == g.num_nodes()
    # new_g = DGLHeteroGraph(new_edge_idx, ['_U'], ['_E'])
    new_g.ndata["feat"] = g.ndata["feat"]
    # new_g.ndata['_ID'] = g.ndata['_ID']
    new_g.edata["bases"] = bases
    if "feat" in g.edata.keys():
        edge_attr = g.edata.pop("feat")
        # print(edge_attr)
        edge_attr_dense = to_dense_adj(
            edge_idx, edge_attr=edge_attr, total_edge_index=new_edge_idx
        )
        if len(edge_attr.shape) == 1:
            edge_attr_dense = edge_attr_dense.view(-1)
        else:
            edge_attr_dense = edge_attr_dense.view(-1, edge_attr.shape[-1])
        # assert (len(edge_attr_dense.shape) == 2)
        assert bases.shape[0] == edge_attr_dense.shape[0]
        new_g.edata["feat"] = edge_attr_dense
    # new_g.edata['_ID'] = g.edata['_ID']
    # print(new_g)
    if basis_cache is not None:
        basis_cache[key] = new_g
    return new_g


class PDFConv(nn.Module):
    def __init__(self, hidden_size):
        super(PDFConv, self).__init__()
        self.pre_ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            torch.nn.GELU(),
        )

        self.ffn = nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.GELU(),
        )

    def forward(self, graph, x_feat, edge_attr, bases):
        with graph.local_scope():
            graph.ndata["x"] = x_feat
            graph.edata["e"] = edge_attr
            graph.apply_edges(fn.u_add_e("x", "e", "pos_e"))
            graph.edata["v"] = self.pre_ffn(graph.edata["pos_e"]) * bases
            graph.update_all(fn.copy_e("v", "pre_aggr"), fn.sum("pre_aggr", "aggr"))
            y = graph.ndata["aggr"]
            # x = x_feat + y
            # y = self.ffn(x)
            # x = x + y
            return self.ffn(y)
