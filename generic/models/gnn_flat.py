import torch

from generic.models.graph_conv import GraphConv
import torch_geometric
from generic.models.g2 import G2Merge
from generic.models.all_pooler import AllPooler, AllPooler2

# class AllPooler(torch.nn.Module):
#     def __init__(
#         self, hidden_dim, num_heads, activation, n_mlp_layers, gconv_activation
#     ):
#         super().__init__()
#         self.conv = GraphConv(
#             hidden_dim,
#             hidden_dim,
#             num_heads=num_heads,
#             gconv_activation=gconv_activation,
#             activation=activation,
#             n_mlp_layers=n_mlp_layers,
#         )

#     def forward(self, feats):
#         nl = feats.shape[0]
#         nnodes = feats.shape[1]
#         hd = feats.shape[2]
#         dst = torch.tensor(list(range(nnodes)) * nl)
#         src = torch.arange(nl * nnodes)
#         new_index = torch.stack([src, dst]).to(feats.device)
#         y = self.conv.forward_nog(feats.reshape((-1, hd)), new_index, None, None)
#         return y


class GnnFlat(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_layers,
        n_mlp_layers,
        layer_pooling,
        n_attention_heads,
        normalize,
        activation,
        residual,
        checkpoint,
        gconv_activation,
        shared_layers,
        g2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_pooling = layer_pooling
        self.n_layers = n_layers
        self.normalize = normalize
        self.dropout = False
        self.residual = residual
        self.checkpoint = checkpoint
        self.g2 = g2

        if self.layer_pooling == "all":
            self.all_pooler = AllPooler2(
                False,
                self.hidden_dim,
                n_attention_heads,
                activation,
                n_mlp_layers,
                gconv_activation,
            )

        self.features_extractors = torch.nn.ModuleList()
        if self.g2:
            self.g2_layers = torch.nn.ModuleList()

        self.edge_reembed = torch.nn.ModuleList()

        if self.normalize:
            self.norms = torch.nn.ModuleList()
        if self.dropout:
            self.drop = torch.nn.Dropout(0.05)

        self.norm_feat = torch_geometric.nn.norm.LayerNorm(
            self.hidden_dim,
            mode="graph",
            affine=False,
        )

        self.shared_layers = shared_layers

        if shared_layers:
            if self.normalize:
                self.norm = torch_geometric.nn.norm.LayerNorm(
                    self.hidden_dim,
                    mode="graph",
                    affine=False,
                )

            self.features_extractor = GraphConv(
                self.hidden_dim,
                self.hidden_dim,
                num_heads=n_attention_heads,
                activation=activation,
                n_mlp_layers=n_mlp_layers,
                gconv_activation=gconv_activation,
            )

            if self.g2:
                self.g2_layer = G2Merge(
                    GraphConv(
                        self.hidden_dim,
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                        activation=activation,
                        n_mlp_layers=n_mlp_layers,
                        gconv_activation=gconv_activation,
                    ),
                )

            self.edge_reembed = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        else:
            self.features_extractors = torch.nn.ModuleList()
            if self.g2:
                self.g2_layers = torch.nn.ModuleList()
            self.edge_reembed = torch.nn.ModuleList()
            if self.normalize:
                self.norms = torch.nn.ModuleList()

            for layer in range(self.n_layers):
                if self.normalize:
                    self.norms.append(
                        torch_geometric.nn.norm.LayerNorm(
                            self.hidden_dim,
                            mode="graph",
                            affine=False,
                        )
                    )

                self.features_extractors.append(
                    GraphConv(
                        self.hidden_dim,
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                        activation=activation,
                        n_mlp_layers=n_mlp_layers,
                        gconv_activation=gconv_activation,
                    )
                )
                if self.g2:
                    self.g2_layers.append(
                        G2Merge(
                            GraphConv(
                                self.hidden_dim,
                                self.hidden_dim,
                                num_heads=n_attention_heads,
                                activation=activation,
                                n_mlp_layers=n_mlp_layers,
                                gconv_activation=gconv_activation,
                            ),
                        )
                    )
                self.edge_reembed.append(
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                )
        # self.setup_un_shared()

    # def norm_shared(self, i, features, batch):
    #     return self.norm(features, batch)

    # def norm_unshared(self, i, features, batch):
    #     return self.norms[i](features, batch)

    # def conv_shared(self, i, g, feats, ef):
    #     return self.features_extractor(g, feats, ef)

    # def conv_unshared(self, i, g, feats, ef):
    #     return self.features_extractors[i](g, feats, ef)

    # def g2l_shared(self, i, fb, f, ei, ef):
    #     return self.g2_layer(fb, f, ei, ef)

    # def g2l_unshared(self, i, fb, f, ei, ef):
    #     return self.g2_layers[i](fb, f, ei, ef)

    # def erl_shared(self, i, ef):
    #     return self.edge_reembed(ef)

    # def erl_unshared(self, i, ef):
    #     return self.edge_reembed[i](ef)

    # def setup_un_shared(self):
    #     if self.shared_layers:
    #         self.erl = self.erl_shared
    #         self.g2l = self.g2l_shared
    #         self.conv = self.conv_shared
    #         self.norm = self.norm_shared
    #     else:
    #         self.erl = self.erl_unshared
    #         self.g2l = self.g2l_unshared
    #         self.conv = self.conv_unshared
    #         self.norm = self.norm_unshared

    # @torch.autocast(device_type="cuda")
    def forward(self, g, features, edge_features, norm_mask=None):
        if self.layer_pooling == "all":
            features_list = [features]
        if norm_mask is None:
            norm_mask = torch.arange(features.shape[0], device=features.device)
        features[norm_mask] = self.norm_feat(
            features[norm_mask], batch=g.batch[norm_mask]
        )
        if self.layer_pooling == "all":
            features_list.append(features)

        if self.layer_pooling == "last" and self.residual:
            previous_feat = features

        for layer in range(self.n_layers):
            if not self.shared_layers:
                if self.normalize:
                    norm = self.norms[layer]
                conv = self.features_extractors[layer]
                if self.g2:
                    g2l = self.g2_layers[layer]
                erl = self.edge_reembed[layer]
            else:
                if self.normalize:
                    norm = self.norm
                conv = self.features_extractor
                if self.g2:
                    g2l = self.g2_layer
                erl = self.edge_reembed
            if self.normalize and layer != 0:
                features[norm_mask] = norm(
                    features[norm_mask], batch=g.batch[norm_mask]
                )
            if self.dropout:
                features = self.drop(features)
            if self.g2:
                features_before = features
            if layer % self.checkpoint:
                features, _ = torch.utils.checkpoint.checkpoint(
                    conv,
                    g._graph,
                    features,
                    use_reentrant=False,
                )
            else:
                features = conv(g, features, erl(edge_features))

            if self.residual and not self.g2:
                if self.layer_pooling == "all":
                    features += features_list[-1][:, : self.hidden_dim]
                else:
                    features += previous_feat[:, : self.hidden_dim]

            elif self.g2:
                features = g2l(
                    features_before,
                    features,
                    g.edge_index,
                    erl(edge_features),
                )

            if self.layer_pooling == "all":
                features_list.append(features)
            else:
                previous_feat = features

        if self.layer_pooling == "all":
            return self.all_pooler(
                torch.stack(features_list[2:], dim=0), features_list[0]
            ), None
            # return features_list[1:], None

        return features, None
