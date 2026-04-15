import torch

from generic.models.graph_conv import GraphConv
import torch_geometric
from generic.models.g2 import G2Merge


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

        self.features_extractors = torch.nn.ModuleList()
        if self.g2:
            self.g2_layers = torch.nn.ModuleList()

        if self.normalize:
            self.norms = torch.nn.ModuleList()
        if self.dropout:
            self.drop = torch.nn.Dropout(0.05)

        self.norm_feat = torch_geometric.nn.norm.LayerNorm(
            self.hidden_dim,
            mode="graph",
            affine=False,
        )

        for layer in range(self.n_layers):
            if self.normalize:
                if layer == 0 or not shared_layers:
                    self.norms.append(
                        torch_geometric.nn.norm.LayerNorm(
                            self.hidden_dim,
                            mode="graph",
                            affine=False,
                        )
                    )
                else:
                    self.norms.append(self.norms[0])

            if layer == 0 or not shared_layers:
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

            else:
                self.features_extractors.append(self.features_extractors[0])
                if self.g2:
                    self.g2_layers.append(self.g2_layers[0])

    # @torch.autocast(device_type="cuda")
    def forward(self, g, features, edge_features, norm_mask=None):
        if norm_mask is None:
            norm_mask = torch.arange(features.shape[0], device=features.device)
        features[norm_mask] = self.norm_feat(
            features[norm_mask], batch=g.batch[norm_mask]
        )
        if self.layer_pooling == "all":
            features_list = [features]

        if self.layer_pooling == "last" and self.residual:
            previous_feat = features

        for layer in range(self.n_layers):
            if self.normalize and layer != 0:
                features[norm_mask] = self.norms[layer](
                    features[norm_mask], batch=g.batch[norm_mask]
                )
            if self.dropout:
                features = self.drop(features)
            if self.g2:
                features_before = features
            if layer % self.checkpoint:
                features, _ = torch.utils.checkpoint.checkpoint(
                    self.features_extractors[layer],
                    g._graph,
                    features,
                    use_reentrant=False,
                )
            else:
                features = self.features_extractors[layer](
                    g,
                    features,
                    edge_features,
                )

            # if self.layer_pooling == "all" and not self.g2:

            if self.residual and not self.g2:
                if self.layer_pooling == "all":
                    features += features_list[-1][:, : self.hidden_dim]
                else:
                    features += previous_feat[:, : self.hidden_dim]

            elif self.g2:
                # tau = self.g2_layers[layer](
                #     g, features_before, self.edge_embedders[layer](edge_features)
                # )
                # features = (1 - tau) * features_before + tau * features

                features = self.g2_layers[layer](
                    features_before,
                    features,
                    g.edge_index,
                    edge_features,
                )

                # if self.layer_pooling != "all":
                #     previous_feat = features
                # else:
                #     features_list.append(features)

            if self.layer_pooling == "all":
                features_list.append(features)
            else:
                previous_feat = features

        if self.layer_pooling == "all":
            return features_list[1:], None

        return features, None
