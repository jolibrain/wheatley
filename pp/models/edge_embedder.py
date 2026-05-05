import torch


class EdgeEmbedder(torch.nn.Module):
    def __init__(
        self,
        output_dim,
    ):
        super().__init__()
        self.emb = torch.nn.Embedding(1, output_dim)

    def forward(self, g, eid):
        return self.emb(torch.tensor([0] * len(eid), device=self.emb.weight.device))
