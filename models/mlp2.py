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


class MLP2(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm,
        activation,
    ):
        super(MLP2, self).__init__()
        if n_layers < 1:
            raise Exception("Number of layers must be >= 1")
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.batch_norms = torch.nn.ModuleList()

        if self.n_layers == 1:
            self.layers.append(torch.nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
            for i in range(self.n_layers - 2):
                self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                if self.batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
            self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
            if self.batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(output_dim))

        if type(activation) == str:
            if activation == "tanh":
                self.activation_layer = torch.nn.Tanh()
            elif activation == "relu":
                self.activation_layer = torch.nn.LeakyReLU()
            elif activation == "elu":
                self.activation_layer = torch.nn.ELU()
            elif activation == "gelu":
                self.activation_layer = torch.nn.GELU()
            elif activation == "selu":
                self.activation_layer = torch.nn.SELU()
            else:
                raise Exception("Activation not recognized")
        else:
            self.activation_layer = activation()

    def forward(self, x):
        for layer in range(self.n_layers - 1):
            if self.batch_norm:
                x = self.batch_norms[layer](
                    self.activation_layer(self.layers[layer](x))
                )
            else:
                x = self.activation_layer(self.layers[layer](x))
        return self.layers[-1](x)
