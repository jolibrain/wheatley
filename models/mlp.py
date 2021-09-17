import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        n_layers,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm,
        activation,
        device,
    ):
        super(MLP, self).__init__()
        if n_layers < 3:
            raise Exception("Number of layers must be >= 3")
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.batch_norms = torch.nn.ModuleList()

        for i in range(self.n_layers - 1):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            elif i < self.n_layers - 2:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            if self.batch_norm:
                if i < self.n_layers - 2:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                else:
                    self.batch_norms.append(nn.BatchNorm1d(output_dim))
        if activation == "tanh":
            self.activation_layer = nn.Tanh()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        else:
            raise Exception("Activation not recognized")
        self.to(device)

    def forward(self, x):
        for layer in range(self.n_layers - 2):
            if self.batch_norm:
                x = self.activation_layer(self.batch_norms[layer](self.layers[layer](x)))
            else:
                x = self.activation_layer(self.layers[layer](x))
        return self.layers[self.n_layers - 2](x)
