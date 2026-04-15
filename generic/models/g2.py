import torch
import torch_geometric
from torch_scatter import scatter
from .graph_conv import GraphConv


class G2(torch.nn.Module):
    def __init__(self, conv, p=3):
        super(G2, self).__init__()
        self.conv = conv
        self.p = p
        # TODO : honor activation?

    def forward(self, X, edge_index, edge_features=None):

        # X = torch.nn.functional.elu(self.conv(g, X, edge_features))
        X = self.conv.forward_nog(X, edge_index, edge_features)
        # print(f"abs {torch.abs(X[g.edge_index[0]] - X[g.edge_index[1]]).mean()}")
        gg = torch.tanh(
            scatter(
                (torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                edge_index[0],
                0,
                dim_size=X.size(0),
                reduce="mean",
            )
            + scatter(
                (torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                edge_index[1],
                0,
                dim_size=X.size(0),
                reduce="mean",
            )
        )

        return gg


class G2Merge(torch.nn.Module):
    def __init__(self, conv, p=3):
        super(G2Merge, self).__init__()
        self.g2 = G2(conv, p)

    def forward(self, res, new, edge_index, edge_features):
        tau = self.g2(res, edge_index, edge_features)
        return (1 - tau) * res + tau * new
