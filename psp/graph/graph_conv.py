import torch

# from generic.eegatconv import EEGATConv
from torch_geometric.nn.conv import GATv2Conv
from generic.gcongat import HybridConv_v2
from generic.mlp import MLP


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
        gcon=False,
        n_mlp_layers=None,
        norm=None,
        activation=None,
    ):
        super().__init__()
        self.pyg = pyg
        self.gcon = gcon
        self.gcon_has_mlp = self.gcon and (in_dim != out_dim)
        if self.gcon:
            self.conv = HybridConv_v2(
                out_dim, out_dim, num_heads=num_heads, edge_dim=edge_dim
            )
        elif self.pyg:
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
        if not self.gcon:
            self.mlp = MLP(
                n_layers=n_mlp_layers,
                input_dim=out_dim * num_heads,
                hidden_dim=out_dim * num_heads,
                output_dim=out_dim,
                norm=norm,
                activation=activation,
            )
        if self.gcon_has_mlp:
            self.mlp = MLP(
                n_layers=n_mlp_layers,
                input_dim=in_dim,
                hidden_dim=in_dim,
                output_dim=out_dim,
                norm=norm,
                activation=activation,
            )

    def forward(self, g, node_feats, edge_feats, edge_efeats=None):
        if self.gcon:
            if self.gcon_has_mlp:
                node_feats = self.mlp(node_feats)
            return self.conv(node_feats, g.edge_index, edge_feats), None
        elif self.pyg:
            feats = self.conv(node_feats, g.edge_index, edge_feats)
            return self.mlp(feats), None
        else:
            f, ef = self.conv(g, node_feats, edge_feats, edge_efeats)
            return f.flatten(start_dim=-2, end_dim=-1), ef
            # return self.conv(g, node_feats, edge_feats, edge_efeats)

    def forward_nog(self, node_feats, edge_index, edge_feats, edge_efeats=None):
        if self.gcon:
            if self.gcon_has_mlp:
                node_feats = self.mlp(node_feats)
            return self.conv(node_feats, edge_index, edge_feats)
        else:
            feats = self.conv(node_feats, edge_index, edge_feats)
            return self.mlp(feats)
