#    MIT License
#
#    Copyright (c) Microsoft Corporation.
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE
#
import torch
import torch.nn as nn
from torch.nn import Dropout


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ffn_embedding_dim, activation_fn, activation_dropout, dropout, output_dim=None):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.activation_fn = activation_fn
        self.activation_dropout_module = Dropout(activation_dropout)
        if output_dim is None:
            self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        else:
            self.fc2 = nn.Linear(ffn_embedding_dim, output_dim)
        self.dropout_module = Dropout(dropout)

    def forward(self, x):
        x = self.activation_fn()(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
