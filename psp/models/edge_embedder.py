import torch
from generic.mlp import MLP
from generic.models.graph_conv import GraphConv


class SimpleEdgeEmbedder(torch.nn.Module):
    def __init__(
        self,
        output_dim,
    ):
        super().__init__()
        self.emb = torch.nn.Embedding(1, output_dim)

    def forward(self, g, eid):
        return self.emb(torch.tensor([0] * len(eid), device=self.emb.weight.device))


class RPEdgeEmbedder(torch.nn.Module):
    def __init__(self, output_dim, n_layers, activation):
        super().__init__()
        self.n_feats = 4
        self.mlp = MLP(
            n_layers,
            self.n_feats,
            output_dim,
            output_dim,
            False,
            activation,
        )

    def forward(self, g, eid):
        x = g.r[eid]

        return self.mlp(x)


class UseEdgeEmbedder(torch.nn.Module):
    def __init__(self, output_dim, n_layers, activation):
        super().__init__()
        self.n_feats = 1
        self.mlp = MLP(
            n_layers,
            self.n_feats,
            output_dim,
            output_dim,
            False,
            activation,
        )

    def forward(self, g, eid):
        x = g.att_uses[eid].unsqueeze(-1)

        return self.mlp(x)
