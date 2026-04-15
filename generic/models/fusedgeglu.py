import torch


class FusedGeGLU(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim):
        super().__init__()
        self.dim = hidden_dim
        self.num_heads = num_heads
        self.lin_before_x = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin_before_y = torch.nn.Linear(hidden_dim, hidden_dim)
        if self.num_heads > 1:
            self.lin = torch.nn.Linear(hidden_dim * num_heads, hidden_dim)
        else:
            self.lin = torch.nn.Identity()

    def forward(self, d):
        x, y = torch.chunk(d, 2, dim=-1)
        x2 = self.lin_before_x(x.view(-1, self.num_heads, self.dim)).view(
            -1, self.dim * self.num_heads
        )
        y2 = self.lin_before_y(y.view(-1, self.num_heads, self.dim)).view(
            -1, self.dim * self.num_heads
        )
        # r = x * torch.nn.functional.gelu(y)
        r = x2 * torch.nn.functional.gelu(y2)
        return self.lin(r)

        # d = self.lin(d)
        # x, y = torch.chunk(d, 2, dim=-1)
        # return x * torch.nn.functional.gelu(y)
