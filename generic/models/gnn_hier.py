import torch

from generic.models.graph_conv import GraphConv
from generic.mlp import MLP
from typing import Callable, Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor, PairTensor, Tensor
import torch_geometric


from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.dense import Linear
from torch_scatter import scatter_max, scatter_min, scatter_add, scatter

from torch_sparse import SparseTensor, remove_diag
from torch.distributions.bernoulli import Bernoulli
from generic.models.g2 import G2Merge

from generic.models.all_pooler import AllPooler2

Scorer = Callable[[Tensor, Adj, OptTensor, OptTensor], Tensor]
torch._dynamo.config.capture_scalar_outputs = True


def check_consistency(adj, batch):
    if isinstance(adj, SparseTensor):
        row, col, _ = adj.coo()
    else:
        row = adj[0]
        col = adj[1]

    for i in range(row.shape[0]):
        if batch[row[i]] != batch[col[i]]:
            print("INCONSISTENCY at elt ", i)


class ScoreAggr(torch.nn.Module):
    # or MLP ?
    def __init__(self, hidden_dim):
        super(ScoreAggr, self).__init__()
        self.lin = torch.nn.Linear(hidden_dim + 1, hidden_dim)
        # self.mlp = MLP(3, hidden_dim + 1, hidden_dim + 1, hidden_dim, False, "gelu")

    def forward(self, features, score):
        # return self.mlp(torch.cat((features, score), dim=-1))
        return self.lin(torch.cat((features, score), dim=-1))


class NodeAggr(torch.nn.Module):
    def __init__(
        self, hidden_dim, num_heads, n_mlp_layers, activation, gconv_activation
    ):
        super(NodeAggr, self).__init__()
        # self.emb = torch.nn.Embedding(1, hidden_dim)
        self.gc = GraphConv(
            hidden_dim,
            hidden_dim,
            num_heads=num_heads,
            activation=activation,
            gconv_activation=gconv_activation,
            n_mlp_layers=n_mlp_layers,
            add_self_loops=True,
        )

    def forward(self, x, index, mis):
        # below merge nodes into pool node
        # pool_nodes_data = self.emb(torch.tensor([0] * dim_size, device=x.device))
        # x2 = torch.cat([pool_nodes_data, x])

        # new_index = torch.stack(
        #     [
        #         torch.tensor(
        #             list(range(dim_size, dim_size + x.shape[0])), device=x.device
        #         ),
        #         index,
        #     ]
        # )
        # y = self.gc.forward_nog(x2, new_index, None, None)
        # return y[:dim_size]

        # new_index = torch.stack(
        #     [torch.tensor(list(range(x.shape[0])), device=x.device), index]
        # )
        # y = self.gc.forward_nog(x, new_index, None, None)
        # return y[:dim_size]
        dim_size = mis.sum()

        # SCATTER
        # cluster_x = scatter(x, index, dim=0, dim_size=dim_size, reduce="mean")
        # newx = torch.cat([cluster_x, x], dim=0)
        # new_index = torch.stack(
        #     [torch.arange(x.size(0), device=x.device) + dim_size, index]
        # )

        # BELOW MERGE ONLY DISCARDED, SOURCE CLUSTER CENTERS as SELFLOOPS
        cluster_x = x[mis]
        discarded_x = x[torch.logical_not(mis)]
        newx = torch.cat([cluster_x, discarded_x], dim=0)

        new_index = torch.stack(
            [
                torch.arange(discarded_x.size(0), device=x.device) + dim_size,
                index[torch.logical_not(mis)],
            ]
        )
        y = self.gc.forward_nog(newx, new_index, None, None)
        return y[:dim_size]


class EdgeScorer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeScorer, self).__init__()
        # self.lin = Linear(hidden_dim, 1)

    def forward(self, x):
        # return self.lin(x).sigmoid()
        return x[..., [0]]


class EdgePreScorer(torch.nn.Module):
    def __init__(
        self, hidden_dim, num_heads, n_mlp_layers, activation, gconv_activation
    ):
        super(EdgePreScorer, self).__init__()

        self.norm_ea = torch_geometric.nn.norm.LayerNorm(
            3 * hidden_dim, mode="graph", affine=False
        )
        self.lin = Linear(hidden_dim * 3, hidden_dim)
        # self.mlp = MLP(3, hidden_dim * 3, hidden_dim * 3, hidden_dim, False, "geglu")

    def forward(self, x, edge_index, edge_features, batch):
        edge_batch = batch[edge_index[0]]
        edge_attr = torch.cat(
            [x[edge_index[0]], x[edge_index[1]], edge_features], dim=-1
        )

        edge_attr = self.norm_ea(edge_attr, batch=edge_batch)
        scored = self.lin(edge_attr)
        # scored = self.mlp(edge_attr)
        return scored


class EdgePostScorer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(EdgePostScorer, self).__init__()

        self.lin = Linear(hidden_dim + 1, hidden_dim)

    def forward(self, edge_feats, score):
        return self.lin(torch.cat([edge_feats, score.unsqueeze(-1)], dim=-1))


class GraphPool(torch.nn.Module):
    def __init__(
        self, hidden_dim, num_heads, activation, gconv_activation, n_mlp_layers
    ):
        super(GraphPool, self).__init__()
        self.emb = torch.nn.Embedding(1, hidden_dim)
        self.gc = GraphConv(
            hidden_dim,
            hidden_dim,
            num_heads=num_heads,
            activation=activation,
            gconv_activation=gconv_activation,
            n_mlp_layers=n_mlp_layers,
        )
        self.norm = torch_geometric.nn.norm.LayerNorm(
            hidden_dim, mode="graph", affine=True
        )

    def forward(self, x, batch):
        nbatch = batch.max().item() + 1
        poolnodes = self.emb(torch.tensor([0] * nbatch, device=x.device))
        x = self.norm(x, batch=batch)
        x2 = torch.cat([poolnodes, x])
        index = torch.stack(
            [
                torch.tensor(list(range(nbatch, nbatch + x.shape[0])), device=x.device),
                batch,
            ]
        )
        y = self.gc.forward_nog(x2, index, None, None)
        return y[:nbatch]


class EdgeAggr(torch.nn.Module):
    def __init__(
        self, hidden_dim, num_heads, n_mlp_layers, activation, gconv_activation
    ):
        super(EdgeAggr, self).__init__()
        self.emb = torch.nn.Embedding(2, hidden_dim)
        self.gc = GraphConv(
            hidden_dim,
            hidden_dim,
            num_heads=num_heads,
            activation=activation,
            gconv_activation=gconv_activation,
            n_mlp_layers=n_mlp_layers,
        )

    def forward(self, x, row, col, c, batch):
        flattenedei = row * c + col
        u, uinv = torch.unique(flattenedei, return_inverse=True)
        pool_nodes_data = self.emb(torch.tensor([0] * u.shape[0], device=x.device))
        # pool_nodes_data2 = self.emb(torch.tensor([1] * u.shape[0], device=x.device))
        # x2 = torch.cat([pool_nodes_data, pool_nodes_data2, x, x])
        x2 = torch.cat([pool_nodes_data, x])
        output = uinv
        # output = torch.cat([uinv, uinv + uinv.shape[0]])
        new_index = torch.stack(
            [
                torch.tensor(
                    list(range(u.shape[0], u.shape[0] + x.shape[0])),
                    device=x.device,
                    # list(range(u.shape[0] * 2, u.shape[0] * 2 + 2 * x.shape[0])),
                    # device=x.device,
                ),
                # uinv,
                output,
            ]
        )
        y = self.gc.forward_nog(x2, new_index, None, None)

        # adj = SparseTensor(
        #     row=u.div(c, rounding_mode="floor"),
        #     col=torch.remainder(u, c),
        #     value=y3,
        #     sparse_sizes=(c, c),
        #     is_sorted=False,
        # )
        # return adj
        row = u.div(c, rounding_mode="floor")
        col = torch.remainder(u, c)
        # edge_index = torch.stack([torch.cat([row, row]), torch.cat([col, col])])
        edge_index = torch.stack([row, col])
        return edge_index, y[: u.shape[0]]
        # return edge_index, y[: u.shape[0] * 2]


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
    edge_index_for_mis: Adj, edge_index_for_cluster, k: int = 1, perm: OptTensor = None
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
    mis = maximal_independent_set(edge_index=edge_index_for_mis, k=k, perm=perm)
    n, device = mis.size(0), mis.device
    if isinstance(edge_index_for_cluster, SparseTensor):
        row, col, _ = edge_index_for_cluster.coo()
    else:
        row, col = edge_index_for_cluster[0], edge_index_for_cluster[1]
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
        # BELOW FOR SPECIAL ADJ
        min_rank[mis] = rank_mis
    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)
    return mis, perm[clusters]


class KMISPooling(torch.nn.Module):
    _heuristics = {None, "greedy", "w-greedy"}
    _passthroughs = {None, "before", "after"}
    _scorers = {
        "linear",
        "linearsigmoid",
        "mlp",
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
        score_passthrough: Optional[str] = None,
        aggr_x: Optional[Union[str, Aggregation]] = None,
        aggr_edge: Union[str, Aggregation] = "sum",
        aggr_score: Union[
            torch.nn.Module, Callable[[Tensor, Tensor], Tensor]
        ] = torch.mul,
        remove_self_loops: bool = False,
        edge_pre_score=None,
        edge_post_score=None,
        edge_score=None,
        node_updater=None,
    ) -> None:
        super(KMISPooling, self).__init__()
        assert score_heuristic in self._heuristics, (
            "Unrecognized `score_heuristic` value."
        )
        assert score_passthrough in self._passthroughs, (
            "Unrecognized `score_passthrough` value."
        )
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
        self.edge_pre_score = edge_pre_score
        self.edge_post_score = edge_post_score
        self.edge_score = edge_score
        self.update_edges = True
        self.bernoulli = False
        self.learnable_distance = True
        self.update_nodes = False
        # self.cluster_with_orig_adj = False
        # BELOW SPECIAL ADJ
        self.cluster_with_orig_adj = True

        if self.update_nodes:
            self.gc = node_updater

        if scorer in ["linear", "linearsigmoid"]:
            assert self.score_passthrough is not None, (
                "`'score_passthrough'` must not be `None` when using `'linear'` scorer"
            )
            self.lin = torch.nn.Linear(in_features=in_channels, out_features=1)
        elif scorer == "mlp":
            self.mlp = MLP(
                n_layers=3,
                input_dim=in_channels,
                hidden_dim=in_channels,
                output_dim=1,
                norm=False,
                activation="gelu",
            )

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
        edge_index: Adj = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
    ) -> Tensor:
        if self.scorer == "linear":
            return self.lin(x)
            # return self.lin(x)
        if self.scorer == "linearsigmoid":
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
        if self.scorer == "mlp":
            return self.mlp(x)
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

        # check_consistency(adj, batch)

        if not isinstance(edge_index, SparseTensor):
            adj = SparseTensor.from_edge_index(edge_index, edge_features, (n, n))

        if self.update_edges or self.learnable_distance or self.update_nodes:
            new_edge_features = self.edge_pre_score(
                features, edge_index, edge_features, batch
            )
            edge_score = self.edge_score(new_edge_features).squeeze(-1)
            new_edge_features = self.edge_post_score(new_edge_features, edge_score)

        if self.update_edges:
            adj = SparseTensor.from_edge_index(edge_index, new_edge_features, (n, n))

        if self.update_nodes:
            edge_index, new_edge_features = torch_geometric.utils.add_self_loops(
                edge_index, new_edge_features, fill_value=1
            )
            features = self.gc.forward_nog(features, edge_index, new_edge_features)

        if self.learnable_distance:
            if self.bernoulli:
                distrib = Bernoulli(edge_score.sigmoid())
                to_keep = distrib.sample().to(torch.bool)
            else:
                to_keep = torch.where(edge_score > 0.0)[0]

            new_edge_features = new_edge_features[to_keep]
            ei0 = edge_index[0][to_keep]
            ei1 = edge_index[1][to_keep]
            new_edge_index = torch.stack([ei0, ei1])

            adj_for_mis = SparseTensor.from_edge_index(
                new_edge_index, new_edge_features, (n, n)
            )
        else:
            adj_for_mis = adj

        score = self._scorer(features)
        updated_score = self._apply_heuristic(score, adj_for_mis)
        perm = torch.argsort(updated_score.view(-1), 0, descending=True)

        if self.cluster_with_orig_adj:
            adj_for_cluster = adj
        else:
            adj_for_cluster = adj_for_mis

        # check_consistency(adj_for_mis, batch)
        mis, cluster = maximal_independent_set_cluster(
            adj_for_mis, adj_for_cluster, self.k, perm
        )

        row, col, val = adj.coo()
        # row, col, val = adj_for_mis.fill_diag(0).coo()
        c = mis.sum()
        if val is None:
            print("val is none!!!!")
            val = torch.ones_like(row, dtype=torch.float)
        if isinstance(self.aggr_edge, EdgeAggr):
            # check_consistency(edge_index, batch)

            edge_index, edge_attr = self.aggr_edge(
                val, cluster[row], cluster[col], c, batch
            )
            # below remove intra cluster edges
            # edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
            #     edge_index, edge_attr
            # )
            # check_consistency(edge_index, batch[mis])

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
            x = self.aggr_x(x, cluster, mis)
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
        hidden_dim,
        n_layers,
        n_mlp_layers,
        layer_pooling,
        n_attention_heads,
        normalize,
        activation,
        residual,
        checkpoint,
        graph_pooling,
        gconv_activation="swiglu",
        shared_layers=False,
        g2=False,
    ):
        super(GnnHier, self).__init__()

        self.g2 = g2
        self.hidden_dim = hidden_dim
        self.layer_pooling = "all"
        self.n_layers = n_layers
        self.sum_res = residual
        self.n_attention_heads = n_attention_heads
        self.checkpoint = checkpoint
        self.learned_pool = True  # graph_pooling in ["learn", "learninv"]
        self.gconv_activation = gconv_activation
        self.n_messages = 1

        self.normalize_down = normalize
        self.normalize_up = normalize

        self.shared_layers = shared_layers

        if self.normalize_down:
            self.norms_down = torch.nn.ModuleList()
        if self.normalize_up:
            self.norms_up = torch.nn.ModuleList()

        self.pre_convs = torch.nn.ModuleList()
        self.norms_pre = torch.nn.ModuleList()
        self.down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        self.merge_up = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        self.score_aggr = torch.nn.ModuleList()
        self.x_aggr = torch.nn.ModuleList()
        self.edge_aggr = torch.nn.ModuleList()
        self.edge_pre_score = torch.nn.ModuleList()
        self.edge_post_score = torch.nn.ModuleList()
        self.edge_score = torch.nn.ModuleList()

        self.graph_pool = torch.nn.ModuleList()
        self.norm_pool = torch.nn.ModuleList()

        self.node_updater = torch.nn.ModuleList()

        if self.layer_pooling == "all":
            self.all_pooler_graph = AllPooler2(
                False,
                self.hidden_dim,
                n_attention_heads,
                activation,
                n_mlp_layers,
                gconv_activation,
            )
            self.all_pooler_nodes = AllPooler2(
                False,
                self.hidden_dim,
                n_attention_heads,
                activation,
                n_mlp_layers,
                gconv_activation,
            )

        self.edge_embedder = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        for i in range(self.n_layers):
            if i == 0 or not self.shared_layers:
                self.node_updater.append(
                    GraphConv(
                        self.hidden_dim,
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                        activation=activation,
                        gconv_activation=self.gconv_activation,
                        n_mlp_layers=n_mlp_layers,
                    )
                )

            if i == 0 or not self.shared_layers:
                self.score_aggr.append(ScoreAggr(hidden_dim=self.hidden_dim))

            if i == 0 or not self.shared_layers:
                self.x_aggr.append(
                    NodeAggr(
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                        n_mlp_layers=n_mlp_layers,
                        activation=activation,
                        gconv_activation=self.gconv_activation,
                    )
                )

            if i == 0 or not self.shared_layers:
                self.edge_aggr.append(
                    EdgeAggr(
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                        n_mlp_layers=n_mlp_layers,
                        activation=activation,
                        gconv_activation=self.gconv_activation,
                    ),
                )
                self.edge_pre_score.append(
                    EdgePreScorer(
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                        n_mlp_layers=n_mlp_layers,
                        activation=activation,
                        gconv_activation=self.gconv_activation,
                    )
                )
                self.edge_post_score.append(
                    EdgePostScorer(
                        self.hidden_dim,
                    )
                )
                self.edge_score.append(EdgeScorer(self.hidden_dim))

            if i == 0 or not self.shared_layers:
                self.pools.append(
                    KMISPooling(
                        self.hidden_dim,
                        # k=1 if i < self.n_layers - 1 else 2,
                        k=1,
                        aggr_x=self.x_aggr[i],
                        aggr_score=self.score_aggr[i],
                        scorer="first",
                        # score_passthrough="before",
                        # score_heuristic="greedy",
                        # score_heuristic=None,
                        score_heuristic="w-greedy",
                        aggr_edge=self.edge_aggr[i],
                        remove_self_loops=False,
                        edge_pre_score=self.edge_pre_score[i],
                        edge_post_score=self.edge_post_score[i],
                        edge_score=self.edge_score[i],
                        node_updater=self.node_updater[i],
                    )
                )

        self.norm_feat = torch_geometric.nn.norm.LayerNorm(
            self.hidden_dim, mode="graph", affine=True
        )
        self.norm_last = torch_geometric.nn.norm.LayerNorm(
            self.hidden_dim, mode="graph", affine=False
        )

        for i in range(3):
            self.norms_pre.append(
                torch_geometric.nn.norm.LayerNorm(
                    self.hidden_dim, mode="graph", affine=False
                )
            )
            self.pre_convs.append(
                GraphConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    num_heads=n_attention_heads,
                    activation=activation,
                    n_mlp_layers=n_mlp_layers,
                    n_messages=self.n_messages,
                    gconv_activation=self.gconv_activation,
                )
            )

        for i in range(self.n_layers + 1):
            if self.normalize_down:
                if i == 0 or not self.shared_layers:
                    self.norms_down.append(
                        torch_geometric.nn.norm.LayerNorm(
                            self.hidden_dim, mode="graph", affine=False
                        )
                    )

            if self.normalize_up:
                if i == 0 or not self.shared_layers:
                    self.norms_up.append(
                        torch_geometric.nn.norm.LayerNorm(
                            self.hidden_dim, mode="graph", affine=False
                        )
                    )
            if i == 0 or not self.shared_layers:
                self.down_convs.append(
                    GraphConv(
                        self.hidden_dim,
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                        activation=activation,
                        n_mlp_layers=n_mlp_layers,
                        n_messages=self.n_messages,
                        gconv_activation=self.gconv_activation,
                    )
                )
            if i == 0 or not self.shared_layers:
                self.up_convs.append(
                    GraphConv(
                        self.hidden_dim,
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                        edge_dim=self.hidden_dim,
                        activation=activation,
                        n_mlp_layers=n_mlp_layers,
                        n_messages=self.n_messages,
                        gconv_activation=self.gconv_activation,
                    )
                )
                if self.g2:
                    self.merge_up.append(
                        G2Merge(
                            GraphConv(
                                self.hidden_dim,
                                self.hidden_dim,
                                num_heads=n_attention_heads,
                                activation=activation,
                                n_mlp_layers=n_mlp_layers,
                                gconv_activation=gconv_activation,
                            ),
                        )
                    )

                elif self.sum_res is False:
                    self.merge_up.append(
                        MLP(
                            n_layers=2,
                            input_dim=self.hidden_dim * 2,
                            hidden_dim=self.hidden_dim * 2,
                            output_dim=self.hidden_dim,
                            norm=False,
                            activation="geglu",
                        )
                    )

        if self.learned_pool:
            if self.layer_pooling == "all":
                for i in range(self.n_layers * 2 + 2):
                    if i == 0 or not self.shared_layers:
                        self.graph_pool.append(
                            GraphPool(
                                self.hidden_dim,
                                num_heads=n_attention_heads,
                                activation=activation,
                                gconv_activation=self.gconv_activation,
                                n_mlp_layers=n_mlp_layers,
                            )
                        )
            else:
                self.graph_pool = GraphPool(
                    self.hidden_dim,
                    num_heads=n_attention_heads,
                    activation=activation,
                    gconv_activation=self.gconv_activation,
                    n_mlp_layers=n_mlp_layers,
                )

    def forward(self, g, features, edge_features, nodesid_n):
        edge_features = self.edge_embedder(edge_features)

        edge_index = g.edge_index

        x = features
        batch = g.batch
        gpfeats = self.graph_pool[0](x, batch)
        x = self.norm_feat(x, batch=batch)

        if self.learned_pool and self.layer_pooling == "all":
            graph_pools = []
            gpdata = self.graph_pool[0](x, batch)
            graph_pools.append(gpdata)

        x0 = x

        # for i in range(3):
        #     x = self.pre_convs[i](g, x, edge_features)
        #     x = self.norms_pre[i](x, batch=batch)

        x = self.down_convs[0](g, x, edge_features)

        if self.learned_pool and self.layer_pooling == "all":
            gpdata = (
                self.graph_pool[0](x, batch)
                if self.shared_layers
                else self.graph_pool[1](x, batch)
            )
            graph_pools.append(gpdata)
        xs = [x]
        edge_indices = [edge_index]
        all_edge_features = [edge_features]
        mises = []
        clusters = []
        batches = [batch]
        if self.layer_pooling == "all":
            pooled_layers = [x]
            pooled_layers.extend(
                [torch.zeros_like(x) for _ in range((self.n_layers * 2 + 1))]
            )
        for i in range(1, self.n_layers + 1):
            if i % self.checkpoint:
                x, edge_index, edge_features, batch, mis, cluster, perm = (
                    torch.utils.checkpoint.checkpoint(
                        self.pools[0] if self.shared_layers else self.pools[i - 1],
                        x,
                        edge_index,
                        edge_features,
                        batch,
                        use_reentrant=False,
                    )
                )
                to_call = 0 if self.shared_layers else i
                if self.normalize_down:
                    x = self.norms_down[to_call](x, batch=batch)
                x = torch.utils.checkpoint.checkpoint(
                    self.down_convs[to_call].forward_nog,
                    x,
                    edge_index,
                    edge_features,
                    use_reentrant=False,
                )
            else:
                x, edge_index, edge_features, batch, mis, cluster, perm = self.pools[
                    0 if self.shared_layers else i - 1
                ](x, edge_index, edge_features, batch)
                to_call = 0 if self.shared_layers else i
                if self.normalize_down:
                    x = self.norms_down[to_call](x, batch=batch)
                x = self.down_convs[to_call].forward_nog(x, edge_index, edge_features)
            batches.append(batch)
            clusters.append(cluster)

            if self.learned_pool and self.layer_pooling == "all":
                if self.shared_layers:
                    gpdata = self.graph_pool[0](x, batch)
                else:
                    gpdata = self.graph_pool[i + 1](x, batch)
                graph_pools.append(gpdata)
                # graph_pools.append(torch.zeros_like(gpdata, device=gpdata.device))

            mises.append(mis)

            if self.layer_pooling == "all":
                target = torch.arange(x0.shape[0], device=mises[0].device)
                for k in range(i):
                    target = target[mises[k]]
                pooled_layers[i][target] = x
                # pooled_layers[i][target] = torch.zeros_like(x, device=x.device)

            if i < self.n_layers:
                xs.append(x)
            edge_indices.append(edge_index)
            all_edge_features.append(edge_features)

        for i in range(self.n_layers):
            j = self.n_layers - 1 - i
            res = xs[j]
            # BELOW CODEX SUGGESTION
            up = x[clusters[j]]
            # BELOW USUAL FORMULATION
            # up = torch.zeros_like(res)
            # up[mises[j]] = x

            to_call = 0 if self.shared_layers else i
            if self.sum_res:
                x = res + up
            elif self.g2:
                x = self.merge_up[to_call](
                    res, up, edge_indices[j], all_edge_features[j]
                )
            else:
                x = self.merge_up[to_call](torch.cat([res, up], dim=-1))
            if self.normalize_up:
                x = self.norms_up[to_call](x, batch=batches[j])
            if i % self.checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    self.up_convs[to_call].forward_nog,
                    x,
                    edge_indices[j],
                    all_edge_features[j],
                    use_reentrant=False,
                )
            else:
                x = self.up_convs[to_call].forward_nog(
                    x, edge_indices[j], all_edge_features[j]
                )

            if self.learned_pool and self.layer_pooling == "all":
                if self.shared_layers:
                    gpdata = self.graph_pool[0](x, batches[j])
                else:
                    gpdata = self.graph_pool[self.n_layers + i + 2](x, batches[j])
                graph_pools.append(gpdata)

            if self.layer_pooling == "all":
                target = torch.arange(x0.shape[0], device=mises[0].device)
                for k in range(j):
                    target = target[mises[k]]
                pooled_layers[self.n_layers + i + 1][target] = x

        if self.normalize_up:
            x = self.norms_up[-1](x, batch=batches[0])
        if self.sum_res:
            x = x + x0
        elif self.g2:
            x = self.merge_up[-1](x0, x, edge_indices[0], all_edge_features[0])
        else:
            x = self.merge_up[-1](torch.cat((x0, x), dim=-1))
        # x = self.norm_last(x, batch=batches[0])
        x = self.up_convs[-1].forward_nog(x, edge_indices[0], all_edge_features[0])

        if self.layer_pooling == "all":
            pooled_layers[-1] = x
            gpdata = self.graph_pool[-1](x, batches[0])
            graph_pools.append(gpdata)
            pn = self.all_pooler_nodes(
                torch.stack(pooled_layers, dim=0),
                features,
            )
            pg = self.all_pooler_graph(torch.stack(graph_pools, dim=0), gpfeats)
            return pn, pg
            # return pooled_layers, graph_pools
        else:
            gpdata = self.graph_pool(x, batches[0])
            return x, gpdata

        return x, None
