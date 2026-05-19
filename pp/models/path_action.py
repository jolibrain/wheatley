import torch
from generic.mlp import MLP


class PathAction(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        input_dim,
        hidden_dim,
        output_dim,
        norm,
        activation,
    ):
        super(PathAction, self).__init__()
        self.mlp = MLP(n_layers, input_dim, hidden_dim, output_dim, norm, activation)

    def forward(self, x):
        x = self.mlp(x)
        var = torch.nn.functional.sigmoid(x[..., 1])
        mean = torch.tanh(x[..., 0])
        # mean = x[..., 0]
        return torch.stack([mean, var], dim=-1)
