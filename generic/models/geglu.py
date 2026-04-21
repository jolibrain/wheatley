import torch


class GeGLU(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = torch.nn.Linear(dim, 2 * dim)

    def forward(self, d):
        d = self.lin(d)
        x, y = torch.chunk(d, 2, dim=-1)
        return x * torch.nn.functional.gelu(y)
