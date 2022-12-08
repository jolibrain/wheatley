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
        if n_layers < 1:
            raise Exception("Number of layers must be >= 1")
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.batch_norms = torch.nn.ModuleList()

        if self.n_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for i in range(self.n_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            if self.batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(output_dim))

        if type(activation) == str:
            if activation == "tanh":
                self.activation_layer = nn.Tanh()
            elif activation == "relu":
                self.activation_layer = nn.LeakyReLU()
            elif activation == "elu":
                self.activation_layer = nn.ELU()
            elif activation == "gelu":
                self.activation_layer = nn.GELU()
            elif activation == "selu":
                self.activation_layer = nn.SELU()
            else:
                raise Exception("Activation not recognized")
        else:
            self.activation_layer = activation()
        self.to(device)

    def forward(self, x):
        for layer in range(self.n_layers - 1):
            if self.batch_norm:
                x = self.batch_norms[layer](self.activation_layer(self.layers[layer](x)))
            else:
                x = self.activation_layer(self.layers[layer](x))
        return self.layers[-1](x)
