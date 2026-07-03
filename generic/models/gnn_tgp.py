import torch
from torch_geometric.loader import DataLoader
import torch_geometric

from tgp.poolers import BNPool
from generic.models.graph_conv import GraphConv
from generic.models.g2 import G2Merge
from torch_geometric.utils import dense_to_sparse


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


class GnnTGP(torch.nn.Module):
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
        super(GnnTGP, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_attention_heads = n_attention_heads
        self.g2 = g2
        self.sum_res = residual
        self.graph_pooling = graph_pooling
        self.layer_pooling = layer_pooling
        self.learned_pool = True
        self.n_layers = n_layers

        self.conv_enc = GraphConv(
            self.hidden_dim,
            self.hidden_dim,
            num_heads=n_attention_heads,
            activation=activation,
            gconv_activation=gconv_activation,
            n_mlp_layers=n_mlp_layers,
        )
        self.conv_pool = GraphConv(
            self.hidden_dim,
            self.hidden_dim,
            num_heads=n_attention_heads,
            activation=activation,
            gconv_activation=gconv_activation,
            n_mlp_layers=n_mlp_layers,
        )
        self.conv_up = GraphConv(
            self.hidden_dim,
            self.hidden_dim,
            num_heads=n_attention_heads,
            activation=activation,
            gconv_activation=gconv_activation,
            n_mlp_layers=n_mlp_layers,
        )
        self.conv_dec = GraphConv(
            self.hidden_dim,
            self.hidden_dim,
            num_heads=n_attention_heads,
            activation=activation,
            gconv_activation=gconv_activation,
            n_mlp_layers=n_mlp_layers,
        )
        self.merge_up = G2Merge(
            GraphConv(
                self.hidden_dim,
                self.hidden_dim,
                num_heads=n_attention_heads,
                activation=activation,
                n_mlp_layers=n_mlp_layers,
                gconv_activation=gconv_activation,
            ),
        )

        self.edge_weighter = torch.nn.Linear(hidden_dim, 1)
        self.edge_unweighter = torch.nn.Linear(1, hidden_dim)

        self.pool = BNPool(
            in_channels=hidden_dim,
            k=1000,
            alpha_DP=1.0,
            K_var=1.0,
            K_mu=10.0,
            K_init=1.0,
            eta=1.0,
            train_K=True,
            act=None,
            dropout=0.0,
            remove_self_loops=True,
            degree_norm=True,
            edge_weight_norm=False,
            adj_transpose=True,
            lift="precomputed",
            s_inv_op="transpose",
            batched=True,
            sparse_output=True,
            cache_preprocessing=False,
            num_neg_samples=None,
        )

        self.graph_pool = GraphPool(
            self.hidden_dim,
            num_heads=n_attention_heads,
            activation=activation,
            gconv_activation=gconv_activation,
            n_mlp_layers=n_mlp_layers,
        )
        self.norm_feat = torch_geometric.nn.norm.LayerNorm(
            self.hidden_dim, mode="graph", affine=True
        )

    def forward(self, g, features, edge_features, nodes_idn):
        edge_index = g.edge_index

        x = features
        batch = g.batch
        gpfeats = self.graph_pool(x, batch)
        x = self.norm_feat(x, batch=batch)
        if self.learned_pool and self.layer_pooling == "all":
            graph_pools = []
            gpdata = self.graph_pool(x, batch)
            graph_pools.append(gpdata)

        x0 = x
        xs = [x]
        edge_index0 = edge_index
        edge_features0 = edge_features

        x = self.conv_enc(g, x, edge_features)
        if self.learned_pool and self.layer_pooling == "all":
            gpdata = self.graph_pool(x, batch)
            graph_pools.append(gpdata)
            pooled_layers = [x]

        pool_out = []

        for i in range(self.n_layers):
            pool_out.append(
                self.pool(
                    x=x,
                    adj=edge_index if i == 0 else pool_out[i - 1].edge_index,
                    edge_weight=self.edge_weighter(edge_features)
                    if i == 0
                    else pool_out[i - 1].edge_weight,
                    batch=batch if i == 0 else pool_out[i - 1].batch,
                )
            )
            x = self.conv_pool.forward_nog(
                pool_out[i].x,
                pool_out[i].edge_index,
                self.edge_unweighter(pool_out[i].edge_weight.unsqueeze(-1)),
            )

            if self.learned_pool and self.layer_pooling == "all":
                gpdata = self.graph_pool(x, pool_out[i].batch)
                graph_pools.append(gpdata)

            if self.layer_pooling == "all":
                xp = x
                for k in range(i):
                    xp = self.pool(
                        x=xp,
                        so=pool_out[i - k].so,
                        lifting=True,
                    )
                    if xp.dim() == 3:
                        xp = xp.squeeze(0)
                pooled_layers.append(xp)

            xs.append(x)

        for i in range(self.n_layers - 1, 1, -1):
            res = xs[i]
            up = self.pool(x=x, so=pool_out[i].so, lifting=True).squeeze(0)
            if self.sum_res:
                x = res + up
            elif self.g2:
                x = self.merge_up(
                    res,
                    up,
                    pool_out[i - 1].edge_index,
                    self.edge_unweighter(pool_out[i - 1].edge_weight.unsqueeze(-1)),
                )

            x = self.conv_up.forward_nog(
                x,
                pool_out[i - 1].edge_index,
                self.edge_unweighter(pool_out[i - 1].edge_weight.unsqueeze(-1)),
            )
            if self.learned_pool and self.layer_pooling == "all":
                gpdata = self.graph_pool(x, pool_out[i - 1].batch)
                graph_pools.append(gpdata)

            if self.layer_pooling == "all":
                xp = x
                for k in range(i):
                    xp = self.pool(x=xp, so=pool_out[i - k - 1].so, lifting=True)
                if xp.dim() == 3:
                    xp = x.squeeze(0)
                pooled_layers.append(x)

        res = x0
        up = self.pool(x=x, so=pool_out[1].so, lifting=True)
        if self.sum_res:
            x = res + up
        elif self.g2:
            x = self.merge_up(
                res,
                up,
                pool_out[i - 1].edge_index,
                self.edge_unweighter(pool_out[i - 1].edge_weight).unsqueeze(-1),
            )

        x = self.conv_dec.forward_nog(
            x,
            edge_index0,
            edge_features0,
        )

        if self.layer_pooling == "all":
            pooled_layers[-1] = x
            gpdata = self.graph_pool(x, batch)
            graph_pools.append(gpdata)
            pn = self.all_pooler_nodes(
                torch.stack(pooled_layers, dim=0),
                features,
            )
            pg = self.all_pooler_graph(torch.stack(graph_pools, dim=0), gpfeats)
            return pn, pg
            # return pooled_layers, graph_pools
