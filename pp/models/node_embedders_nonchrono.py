import torch
from generic.mlp import MLP


class PoolNodeEmbedder(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # TODO : should be only a param vector
        self.emb = torch.nn.Embedding(1, output_dim)

    def forward(self, g, nid):
        return self.emb(torch.tensor([0] * len(nid), device=self.emb.weight.device))
