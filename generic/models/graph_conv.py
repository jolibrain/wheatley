import torch

# from generic.eegatconv import EEGATConv
from torch_geometric.nn.conv import GATv2Conv
from generic.gatv3_conv import GATv3Conv
from generic.mlp import MLP
from generic.models.gatv2_act import GATv2ActConv
from generic.models.swiglu import SwiGLU
from generic.models.geglu import GeGLU
from generic.models.glu import GLU


class GraphConv(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        bias=False,
        edge_scoring=False,
        gconv_activation="swiglu",
        edge_dim=None,
        activation="",
        n_mlp_layers=3,
        n_messages=1,
        add_self_loops=False,
        dropout=0,
    ):
        super().__init__()
        self.naive = True
        self.gatv3 = False
        if gconv_activation == "swiglu":
            GATactivation = SwiGLU
        elif gconv_activation == "gelu":
            GATactivation = torch.nn.GELU
        elif gconv_activation == "silu":
            GATactivation = torch.nn.SiLU
        elif gconv_activation == "geglu":
            GATactivation = GeGLU
        elif gconv_activation == "glu":
            GATactivation = GLU
        else:
            GATactivation = torch.nn.LeakyReLU

        self.conv = torch.nn.ModuleList()
        self.mlp = torch.nn.ModuleList()
        self.n_messages = n_messages
        for i in range(n_messages):
            if self.gatv3:
                self.conv.append(
                    GATv3Conv(
                        in_dim,
                        out_dim,
                        num_heads,
                        dropout=0,
                        edge_dim=in_dim if edge_dim is None else edge_dim,
                        bias=bias,
                    )
                )
            elif gconv_activation != "relu":
                self.conv.append(
                    GATv2ActConv(
                        in_dim,
                        out_channels=out_dim,
                        heads=num_heads,
                        add_self_loops=add_self_loops,
                        edge_dim=in_dim if edge_dim is None else edge_dim,
                        bias=bias,
                        activation=GATactivation,
                        dropout=dropout,
                        concat=not self.naive,
                    )
                )
                if not self.naive:
                    self.mlp.append(
                        MLP(
                            n_layers=n_mlp_layers,
                            input_dim=out_dim * num_heads,
                            hidden_dim=out_dim * num_heads,
                            output_dim=out_dim,
                            norm=False,
                            activation=activation,
                        )
                    )

            else:
                self.conv.append(
                    GATv2Conv(
                        in_dim,
                        out_dim,
                        num_heads,
                        add_self_loops=False,
                        edge_dim=in_dim if edge_dim is None else edge_dim,
                        bias=bias,
                        concat=not self.naive,
                    )
                )
                if not self.naive:
                    self.mlp.append(
                        MLP(
                            n_layers=n_mlp_layers,
                            input_dim=out_dim * num_heads,
                            hidden_dim=out_dim * num_heads,
                            output_dim=out_dim,
                            norm=False,
                            activation=activation,
                        )
                    )

    def forward(self, g, node_feats, edge_feats, edge_efeats=None):
        x = node_feats
        for i in range(self.n_messages):
            x = self.conv[i](x, g.edge_index, edge_feats)
            if not self.naive:
                x = self.mlp[i](x)
        return x

    def forward_nog(self, node_feats, edge_index, edge_feats, edge_efeats=None):
        x = node_feats
        for i in range(self.n_messages):
            x = self.conv[i](x, edge_index, edge_feats)
            if not self.naive:
                x = self.mlp[i](x)
        return x
