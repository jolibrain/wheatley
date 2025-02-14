import torch

# from generic.eegatconv import EEGATConv
from torch_geometric.nn.conv import GATv2Conv
from generic.gatv3_conv import GATv3Conv


class GraphConv(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        bias=True,
        edge_scoring=False,
        pyg=False,
        edge_dim=None,
    ):
        super().__init__()
        self.gatv3 = False
        self.pyg = pyg
        if self.pyg:
            if self.gatv3:
                self.conv = GATv3Conv(
                    in_dim,
                    out_dim,
                    num_heads,
                    dropout=0,
                    edge_dim=in_dim if edge_dim is None else edge_dim,
                    bias=bias,
                )
            else:
                self.conv = GATv2Conv(
                    in_dim,
                    out_dim,
                    num_heads,
                    add_self_loops=False,
                    edge_dim=in_dim if edge_dim is None else edge_dim,
                    bias=bias,
                )
        else:
            self.conv = EEGATConv(
                in_dim,
                out_dim,
                in_dim,
                out_dim,
                num_heads=num_heads,
                edge_scoring=edge_scoring,
            )

    def forward(self, g, node_feats, edge_feats, edge_efeats=None):
        if self.pyg:
            return self.conv(node_feats, g.edge_index, edge_feats), None
        else:
            f, ef = self.conv(g, node_feats, edge_feats, edge_efeats)
            return f.flatten(start_dim=-2, end_dim=-1), ef
            # return self.conv(g, node_feats, edge_feats, edge_efeats)

    def forward_nog(self, node_feats, edge_index, edge_feats, edge_efeats=None):
        return self.conv(node_feats, edge_index, edge_feats), None
