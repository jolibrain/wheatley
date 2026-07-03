import torch
from generic.models.graph_conv import GraphConv


class AllPooler(torch.nn.Module):
    def __init__(
        self, hidden_dim, num_heads, activation, n_mlp_layers, gconv_activation
    ):
        super().__init__()
        self.conv = GraphConv(
            hidden_dim,
            hidden_dim,
            num_heads=num_heads,
            gconv_activation=gconv_activation,
            activation=activation,
            n_mlp_layers=n_mlp_layers,
        )
        self.emb = torch.nn.Embedding(1, hidden_dim)

    def forward(self, feats):
        nl = feats.shape[0]
        nnodes = feats.shape[1]
        hd = feats.shape[2]
        poolnodes = self.emb(torch.tensor([0] * nnodes, device=feats.device))
        feats2 = torch.cat([poolnodes, feats.reshape((-1, hd))])

        dst = torch.tensor(list(range(nnodes)) * nl, device=feats.device)
        src = torch.arange(nl * nnodes, device=feats.device) + nnodes
        new_index = torch.stack([src, dst])
        y = self.conv.forward_nog(feats2, new_index, None, None)
        return y[:nnodes]


class AllPooler2(torch.nn.Module):
    def __init__(
        self, on_emb, hidden_dim, num_heads, activation, n_mlp_layers, gconv_activation
    ):
        super().__init__()
        self.conv = GraphConv(
            hidden_dim,
            hidden_dim,
            num_heads=num_heads,
            gconv_activation=gconv_activation,
            activation=activation,
            n_mlp_layers=n_mlp_layers,
        )
        self.on_emb = on_emb
        if self.on_emb:
            self.emb = torch.nn.Embedding(1, hidden_dim)

    def forward(self, feats, poolfeats=None):
        nl = feats.shape[0]
        nnodes = feats.shape[1]
        hd = feats.shape[2]
        if poolfeats is None:
            if self.on_emb:
                poolfeats = self.emb(
                    torch.zeros(nnodes, dtype=torch.long, device=feats.device)
                )
            else:
                poolfeats = torch.zeros_like(feats[0], device=feats.device)
        feats = torch.cat([poolfeats.unsqueeze(0), feats], dim=0)
        src = torch.arange((nl) * nnodes) + nnodes
        dst = torch.tensor(list(range(nnodes)) * (nl))
        new_index = torch.stack([src, dst]).to(feats.device)
        y = self.conv.forward_nog(feats.reshape((-1, hd)), new_index, None, None)
        return y[:nnodes]
