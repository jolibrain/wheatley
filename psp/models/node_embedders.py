import torch
from generic.mlp import MLP


class SimpleNodeEmbedder(torch.nn.Module):
    def __init__(
        self,
        output_dim,
        n_layers,
        activation,
        lappe=None,
        rwpe=None,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.nfeats = 15
        self.lappe = lappe if lappe is not None else 0
        self.rwpe = rwpe if rwpe is not None else 0
        self.mlp = MLP(
            n_layers,
            self.nfeats + self.lappe + self.rwpe,
            output_dim,
            output_dim,
            False,
            activation,
        )

    def forward(self, g, nid):
        x = torch.empty(
            (nid.shape[0], self.nfeats + self.lappe + self.rwpe),
            device=self.mlp.layers[0].weight.device,
        )

        x[:, 0] = g.affected[nid]
        x[:, 1] = g.selectable[nid]
        x[:, 2] = g.job[nid]
        x[:, 3:6] = g.normalized_durations[nid]
        x[:, 6:9] = g.normalized_tct[nid]
        x[:, 9:12] = g.normalized_tardiness[nid]
        x[:, 12] = g.has_due_date[nid]
        x[:, 13] = g.normalized_due_dates[nid]
        x[:, 14] = g.past[nid]

        if self.lappe != 0:
            x[:, self.nfeats : self.nfeats + self.lappe] = g.laplacian_eigenvector_pe[
                nid
            ]
        if self.rwpe != 0:
            x[:, self.nfeats + self.lappe : self.nfeats + self.lappe + self.rwpe] = (
                g.random_walk_pe[nid]
            )
        return self.mlp(x)


class PoolNodeEmbedder(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.emb = torch.nn.Embedding(1, output_dim)

    def forward(self, g, nid):
        return self.emb(torch.tensor([0] * len(nid), device=self.emb.weight.device))


class ResNodeEmbedder(torch.nn.Module):
    def __init__(self, output_dim, max_n_resources):
        super().__init__()
        self.emb = torch.nn.Embedding(max_n_resources, output_dim)

    def forward(self, g, nid):
        return self.emb(g.resource_id[nid])


class ResCalNodeEmbedder(torch.nn.Module):
    def __init__(self, id_size, output_dim, max_n_resources):
        super().__init__()
        self.emb = torch.nn.Embedding(max_n_resources, id_size)
        self.output_dim = output_dim
        self.res_cal_emb = MLP(
            2,
            3,
            output_dim - id_size,
            output_dim - id_size,
            False,
            "gelu",
        )

    def forward(self, g, nid):
        out = torch.empty(nid, self.output_dim, device=self.emb.device)
        out[nid, 0:2] = self.emb(g.resource_id[nid])
        out[nid, 2:] = self.res_cal_emb(g.res_cal[g.resource_id[nid]])
        return out
