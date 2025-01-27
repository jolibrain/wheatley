#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#

import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        input_dim,
        hidden_dim,
        output_dim,
        norm,
        activation,
    ):
        super(MLP, self).__init__()
        if n_layers < 1:
            raise Exception("Number of layers must be >= 1")
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.norm = norm

        if self.norm:
            self.norms = torch.nn.ModuleList()

        last_dim = input_dim
        for layer_id in range(self.n_layers - 1):
            self.layers.append(torch.nn.Linear(last_dim, hidden_dim))
            if self.norm:
                self.norms.append(torch.nn.LayerNorm(hidden_dim))

            last_dim = hidden_dim

        self.layers.append(torch.nn.Linear(last_dim, output_dim))

        if type(activation) == str:
            act_map = {
                "tanh": torch.nn.Tanh,
                "relu": torch.nn.LeakyReLU,
                "elu": torch.nn.ELU,
                "gelu": torch.nn.GELU,
                "selu": torch.nn.SELU,
                "silu": torch.nn.SiLU,
            }
            activation = act_map[activation]

        self.activation_layer = activation()

    def forward(self, x):
        for layer in range(self.n_layers - 1):
            x = self.layers[layer](x)
            if self.norm:
                x = self.norms[layer](x)
            x = self.activation_layer(x)

        return self.layers[-1](x)
