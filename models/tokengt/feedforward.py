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
