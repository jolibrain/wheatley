import torch

from psp.graph.graph_conv import GraphConv
from generic.mlp import MLP
from typing import Callable, Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor, PairTensor, Tensor
import torch_geometric

Scorer = Callable[[Tensor, Adj, OptTensor, OptTensor], Tensor]
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.dense import Linear
from torch_scatter import scatter_max, scatter_min, scatter_add, scatter

from torch_sparse import SparseTensor, remove_diag


class ScoreAggr(torch.nn.Module):
    # or MLP ?
    def __init__(self, hidden_dim):
        super(ScoreAggr, self).__init__()
        self.lin = torch.nn.Linear(hidden_dim + 1, hidden_dim)
        # self.mlp = mlp quite bad results

    def forward(self, features, score):
        # return self.mlp(torch.cat((features, score), dim=-1))
        return self.lin(torch.cat((features, score), dim=-1))


class NodeAggr(torch.nn.Module):
    def __init__(
        self, hidden_dim, num_heads, n_mlp_layers, normalize, activation, gcon
    ):
        super(NodeAggr, self).__init__()
        self.emb = torch.nn.Embedding(1, hidden_dim)
        self.gc = GraphConv(
            hidden_dim,
            hidden_dim,
            num_heads=num_heads,
            edge_scoring=False,
            pyg=True,
            gcon=gcon,
            n_mlp_layers=n_mlp_layers,
            norm=normalize,
            activation=activation,
        )

    def forward(self, x, index, dim_size):
        pool_nodes_data = self.emb(torch.tensor([0] * dim_size, device=x.device))
        x2 = torch.cat([pool_nodes_data, x])

        new_index = torch.stack(
            [
                torch.tensor(
                    list(range(dim_size, dim_size + x.shape[0])), device=x.device
                ),
                index,
            ]
        )
        y = self.gc.forward_nog(x2, new_index, None, None)
        y3 = y[:dim_size]
        return y3


class EdgeAggr(torch.nn.Module):
    def __init__(
        self, hidden_dim, num_heads, n_mlp_layers, normalize, activation, gcon
    ):
        super(EdgeAggr, self).__init__()
        self.emb = torch.nn.Embedding(1, hidden_dim)
        self.gc = GraphConv(
            hidden_dim,
            hidden_dim,
            num_heads=num_heads,
            edge_scoring=False,
            pyg=True,
            gcon=gcon,
            n_mlp_layers=n_mlp_layers,
            norm=normalize,
            activation=activation,
        )

    def forward(self, x, row, col, c):
        flattenedei = row * c + col
        u, uinv = torch.unique(flattenedei, return_inverse=True)
        pool_nodes_data = self.emb(torch.tensor([0] * u.shape[0], device=x.device))
        x2 = torch.cat([pool_nodes_data, x])
        new_index = torch.stack(
            [
                torch.tensor(
                    list(range(u.shape[0], u.shape[0] + x.shape[0])), device=x.device
                ),
                uinv,
            ]
        )
        y = self.gc.forward_nog(x2, new_index, None, None)
        y3 = y[: u.shape[0]]

        row = u.div(c, rounding_mode="floor")
        col = torch.remainder(u, c)
        edge_index = torch.stack([row, col])
        return edge_index, y3


def maximal_independent_set(
    edge_index: Adj, k: int = 1, perm: OptTensor = None
) -> torch.Tensor:
    r"""Returns a Maximal :math:`k`-Independent Set of a graph, i.e., a set of
    nodes (as a :class:`ByteTensor`) such that none of them are :math:`k`-hop
    neighbors, and any node in the graph has a :math:`k`-hop neighbor in the
    returned set.
    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method follows `Blelloch's Alogirithm
    <https://arxiv.org/abs/1202.3205>`_ for :math:`k = 1`, and its
    generalization by `Bacciu et al. <https://arxiv.org/abs/2208.03523>`_ for
    higher values of :math:`k`.
    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: :class:`ByteTensor`
    """
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        device = edge_index.device()
        n = edge_index.size(0)
    else:
        row, col = edge_index[0], edge_index[1]
        device = row.device
        n = edge_index.max().item() + 1
    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)
    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    min_rank = rank.clone()
    while not mask.all():
        for _ in range(k):
            min_neigh = torch.full_like(min_rank, fill_value=n)
            scatter_min(min_rank[row], col, out=min_neigh)
            torch.minimum(min_neigh, min_rank, out=min_rank)
        mis = mis | torch.eq(rank, min_rank)
        mask = mis.clone().byte()
        for _ in range(k):
            max_neigh = torch.full_like(mask, fill_value=0)
            scatter_max(mask[row], col, out=max_neigh)
            torch.maximum(max_neigh, mask, out=mask)
        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n
    return mis


def maximal_independent_set_cluster(
    edge_index: Adj, k: int = 1, perm: OptTensor = None
) -> PairTensor:
    r"""Computes the Maximal :math:`k`-Independent Set (:math:`k`-MIS)
    clustering of a graph, as defined in `"Generalizing Downsampling from
    Regular Data to Graphs" <https://arxiv.org/abs/2208.03523>`_.
    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method returns both the :math:`k`-MIS and the clustering, where the
    :math:`c`-th cluster refers to the :math:`c`-th element of the
    :math:`k`-MIS.
    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: (:class:`ByteTensor`, :class:`LongTensor`)
    """
    mis = maximal_independent_set(edge_index=edge_index, k=k, perm=perm)
    n, device = mis.size(0), mis.device
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index[0], edge_index[1]
    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)
    min_rank = torch.full((n,), fill_value=n, dtype=torch.long, device=device)
    rank_mis = rank[mis]
    min_rank[mis] = rank_mis
    for _ in range(k):
        min_neigh = torch.full_like(min_rank, fill_value=n)
        scatter_min(min_rank[row], col, out=min_neigh)
        torch.minimum(min_neigh, min_rank, out=min_rank)
    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)
    return mis, perm[clusters]


class KMISPooling(torch.nn.Module):
    _heuristics = {None, "greedy", "w-greedy"}
    _passthroughs = {None, "before", "after"}
    _scorers = {
        "linear",
        "random",
        "constant",
        "canonical",
        "first",
        "last",
    }

    def __init__(
        self,
        in_channels: Optional[int] = None,
        k: int = 1,
        scorer: Union[Scorer, str] = "linear",
        score_heuristic: Optional[str] = "greedy",
        score_passthrough: Optional[str] = "before",
        aggr_x: Optional[Union[str, Aggregation]] = None,
        aggr_edge: Union[str, Aggregation] = "sum",
        aggr_score: Union[
            torch.nn.Module, Callable[[Tensor, Tensor], Tensor]
        ] = torch.mul,
        remove_self_loops: bool = False,
    ) -> None:
        super(KMISPooling, self).__init__()
        assert (
            score_heuristic in self._heuristics
        ), "Unrecognized `score_heuristic` value."
        assert (
            score_passthrough in self._passthroughs
        ), "Unrecognized `score_passthrough` value."
        if not callable(scorer):
            assert scorer in self._scorers, "Unrecognized `scorer` value."
        self.k = k
        self.scorer = scorer
        self.score_heuristic = score_heuristic
        self.score_passthrough = score_passthrough
        self.aggr_x = aggr_x
        self.aggr_edge = aggr_edge
        self.aggr_score = aggr_score
        self.remove_self_loops = remove_self_loops
        if scorer == "linear":
            assert self.score_passthrough is not None, (
                "`'score_passthrough'` must not be `None`"
                " when using `'linear'` scorer"
            )
            self.lin = torch.nn.Linear(in_features=in_channels, out_features=1)

    def _apply_heuristic(self, x: Tensor, adj: SparseTensor) -> Tensor:
        if self.score_heuristic is None:
            return x
        row, col, _ = adj.coo()
        x = x.view(-1)
        if self.score_heuristic == "greedy":
            k_sums = torch.ones_like(x)
        else:
            k_sums = x.clone()
        for _ in range(self.k):
            scatter_add(k_sums[row], col, out=k_sums)
        return x / k_sums

    def _scorer(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
    ) -> Tensor:
        if self.scorer == "linear":
            return self.lin(x).sigmoid()
        if self.scorer == "random":
            return torch.rand((x.size(0), 1), device=x.device)
        if self.scorer == "constant":
            return torch.ones((x.size(0), 1), device=x.device)
        if self.scorer == "canonical":
            return -torch.arange(x.size(0), device=x.device).view(-1, 1)
        if self.scorer == "first":
            return x[..., [0]]
        if self.scorer == "last":
            return x[..., [-1]]
        return self.scorer(x, edge_index, edge_attr, batch)

    def forward(
        self,
        features,
        edge_index,
        edge_features,
        batch,
    ) -> Tuple[Tensor, Adj, OptTensor, OptTensor, Tensor, Tensor]:
        """"""

        adj, n = edge_index, features.size(0)
        if not isinstance(edge_index, SparseTensor):
            adj = SparseTensor.from_edge_index(edge_index, edge_features, (n, n))
        score = self._scorer(features, edge_index, edge_features, batch)
        updated_score = self._apply_heuristic(score, adj)
        perm = torch.argsort(updated_score.view(-1), 0, descending=True)
        mis, cluster = maximal_independent_set_cluster(adj, self.k, perm)
        row, col, val = adj.coo()
        c = mis.sum()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        if isinstance(self.aggr_edge, EdgeAggr):
            # adj = self.aggr_edge(val, cluster[row], cluster[col], c)
            edge_index, edge_attr = self.aggr_edge(val, cluster[row], cluster[col], c)
        else:
            adj = SparseTensor(
                row=cluster[row],
                col=cluster[col],
                value=val,
                is_sorted=False,
                sparse_sizes=(c, c),
            ).coalesce(self.aggr_edge)
        if self.remove_self_loops:
            adj = remove_diag(adj)
        if self.score_passthrough == "before":
            x = self.aggr_score(features, score)
        else:
            x = features
        if self.aggr_x is None:
            x = x[mis]
        elif isinstance(self.aggr_x, str):
            x = scatter(x, cluster, dim=0, dim_size=mis.sum(), reduce=self.aggr_x)
        else:
            x = self.aggr_x(x, cluster, dim_size=c)
        if self.score_passthrough == "after":
            x = self.aggr_score(x, score[mis])
        # if isinstance(edge_index, SparseTensor):
        #     edge_index, edge_attr = adj, None
        # else:
        # row, col, edge_attr = adj.coo()
        # edge_index = torch.stack([row, col])
        if batch is not None:
            batch = batch[mis]
        perm = perm[mis]
        return x, edge_index, edge_attr, batch, mis, cluster, perm

    def __repr__(self):
        if self.scorer == "linear":
            channels = f"in_channels={self.lin.in_channels}, "
        else:
            channels = ""
        return f"{self.__class__.__name__}({channels}k={self.k})"


class GnnHier(torch.nn.Module):
    def __init__(
        self,
        input_dim_features_extractor,
        hidden_dim_features_extractor,
        n_layers_features_extractor,
        n_mlp_layers_features_extractor,
        layer_pooling,
        n_attention_heads,
        normalize,
        activation_features_extractor,
        residual,
        rwpe_k,
        rwpe_h,
        update_edge_features,
        update_edge_features_pe,
        shared_conv,
        checkpoint,
        gcon,
    ):
        super(GnnHier, self).__init__()
        self.hidden_dim = hidden_dim_features_extractor
        self.layer_pooling = layer_pooling
        self.rwpe_k = rwpe_k
        self.rwpe_h = rwpe_h
        self.update_edge_features = update_edge_features
        self.update_edge_features_pe = update_edge_features_pe
        self.n_layers_features_extractor = n_layers_features_extractor
        self.shared_conv = shared_conv
        self.gcon = gcon
        self.normalize = normalize
        self.sum_res = False
        self.input_dim_features_extractor = input_dim_features_extractor
        self.n_attention_heads = n_attention_heads
        self.checkpoint = checkpoint

        if self.normalize:
            self.norm1 = torch.nn.LayerNorm(self.hidden_dim)
        self.features_embedder = MLP(
            n_layers=n_mlp_layers_features_extractor,
            input_dim=self.input_dim_features_extractor,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            norm=self.normalize,
            activation=activation_features_extractor,
        )

        self.down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        self.score_aggr = torch.nn.ModuleList()
        self.x_aggr = torch.nn.ModuleList()
        self.edge_aggr = torch.nn.ModuleList()

        if self.shared_conv:
            range_conv = 1
        else:
            range_conv = self.n_layers_features_extractor
        for i in range(range_conv):
            self.score_aggr.append(ScoreAggr(hidden_dim=self.hidden_dim))
            self.x_aggr.append(
                NodeAggr(
                    self.hidden_dim,
                    num_heads=1,
                    n_mlp_layers=n_mlp_layers_features_extractor,
                    normalize=self.normalize,
                    activation=activation_features_extractor,
                    gcon=self.gcon,
                )
            )
            self.edge_aggr.append(
                EdgeAggr(
                    self.hidden_dim,
                    num_heads=1,
                    n_mlp_layers=n_mlp_layers_features_extractor,
                    normalize=self.normalize,
                    activation=activation_features_extractor,
                    gcon=self.gcon,
                )
            )

            self.pools.append(
                KMISPooling(
                    self.hidden_dim,
                    k=1
                    if i < self.n_layers_features_extractor - 1 and not self.shared_conv
                    else 2,
                    aggr_x=self.x_aggr[i],
                    aggr_score=self.score_aggr[i],
                    # below first score , very bad results
                    # scorer="first",
                    # score_passthrough=None,
                    aggr_edge=self.edge_aggr[i],
                )
            )

        if self.shared_conv:
            range_conv2 = 1
        else:
            range_conv2 = self.n_layers_features_extractor + 1
        for i in range(range_conv2):
            self.down_convs.append(
                GraphConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    num_heads=n_attention_heads,
                    edge_scoring=self.update_edge_features,
                    pyg=True,
                    gcon=self.gcon,
                    n_mlp_layers=n_mlp_layers_features_extractor,
                    norm=self.normalize,
                    activation=activation_features_extractor,
                )
            )
            self.up_convs.append(
                GraphConv(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    num_heads=n_attention_heads,
                    edge_scoring=self.update_edge_features,
                    edge_dim=self.hidden_dim,
                    pyg=True,
                    gcon=self.gcon,
                    n_mlp_layers=n_mlp_layers_features_extractor,
                    norm=self.normalize,
                    activation=activation_features_extractor,
                )
            )

    def forward(self, g, features, edge_features, pe):
        # if self.normalize:
        #     features = self.norm0(features)

        if self.layer_pooling == "all":
            pooled_layers = [features]

        features = self.features_embedder(features)
        if self.normalize:
            features = self.norm1(features)

        if self.layer_pooling == "all":
            pooled_layers.append(features)

        edge_index = g._graph.edge_index

        x = features
        # x = torch.nn.functional.dropout(features, p=0.5, training=self.training)
        if isinstance(g._graph, torch_geometric.data.Data):
            batch = None
        else:
            batch = g._graph["n"].batch
        x0 = x
        x = self.down_convs[0](g._graph, x, edge_features)
        xs = [x]
        edge_indices = [edge_index]
        all_edge_features = [edge_features]
        perms = []
        if self.layer_pooling == "all":
            pooled_layers.extend([x] * self.n_layers_features_extractor)
        for i in range(1, self.n_layers_features_extractor + 1):
            if self.shared_conv:
                x, edge_index, edge_features, batch, _, cluster, perm = self.pools[0](
                    x, edge_index, edge_features, batch
                )
                x = self.down_convs[0].forward_nog(x, edge_index, edge_features)
            else:
                if i % self.checkpoint:
                    x, edge_index, edge_features, batch, _, cluster, perm = (
                        torch.utils.checkpoint.checkpoint(
                            self.pools[i - 1],
                            x,
                            edge_index,
                            edge_features,
                            batch,
                            use_reentrant=False,
                        )
                    )
                    x, _ = torch.utils.checkpoint.checkpoint(
                        self.down_convs[i].forward_nog, x, edge_index, edge_features
                    )
                    x = torch.utils.checkpoint.checkpoint(self.down_mlps[i], x)
                else:
                    x, edge_index, edge_features, batch, _, cluster, perm = self.pools[
                        i - 1
                    ](x, edge_index, edge_features, batch)
                    x, _ = self.down_convs[i].forward_nog(x, edge_index, edge_features)
                    x = self.down_mlps[i](x)

                if i < self.n_layers_features_extractor:
                    xs.append(x)
                edge_indices.append(edge_index)
                all_edge_features.append(edge_features)
            perms.append(perm)
        for i in range(self.n_layers_features_extractor):
            j = self.n_layers_features_extractor - 1 - i
            res = xs[j]
            edge_index = edge_indices[j]
            perm = perms[j]
            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            if self.shared_conv:
                x = self.up_convs[0].forward_nog(x, edge_index, all_edge_features[j])
            else:
                if i % self.checkpoint:
                    x, _ = torch.utils.checkpoint.checkpoint(
                        self.up_convs[i].forward_nog,
                        x,
                        edge_index,
                        all_edge_features[j],
                    )
                    x = torch.utils.checkpoint.checkpoint(self.up_mlps[i], x)
                else:
                    x, _ = self.up_convs[i].forward_nog(
                        x, edge_index, all_edge_features[j]
                    )
                    x = self.up_mlps[i](x)
            if self.layer_pooling == "all":
                target = pooled_layers[j + 2]
                for k in range(j):
                    target = target[perms[k]]
                target = x
        x = self.up_convs[-1].forward_nog(
            torch.cat((x, x0), dim=-1), edge_index, all_edge_features[0]
        )
        if self.layer_pooling == "all":
            pooled_layers[-1] = x
            return torch.cat(pooled_layers, axis=1)
        return x
