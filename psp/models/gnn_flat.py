import torch

from psp.graph.graph_conv import GraphConv
from generic.mlp import MLP
from .edge_embedder import PspEdgeEmbedder


class GnnFlat(torch.nn.Module):
    def __init__(
        self,
        input_dim_features_extractor,
        hidden_dim_features_extractor,
        n_layers_features_extractor,
        n_mlp_layers_features_extractor,
        layer_pooling,
        n_attention_heads,
        normalize,
        activation_features_extractor,
        residual,
        rwpe_k,
        rwpe_h,
        update_edge_features,
        update_edge_features_pe,
        shared_conv,
        edge_embedding_flavor,
        max_n_resources,
        add_rp_edges,
        factored_rp,
        pyg,
        checkpoint,
        gcon,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim_features_extractor
        self.layer_pooling = layer_pooling
        self.rwpe_k = rwpe_k
        self.rwpe_h = rwpe_h
        self.update_edge_features = update_edge_features
        self.update_edge_features_pe = update_edge_features_pe
        self.n_layers_features_extractor = n_layers_features_extractor
        self.shared_conv = shared_conv
        self.pyg = pyg
        self.normalize = normalize
        self.residual = residual
        self.input_dim_features_extractor = input_dim_features_extractor
        self.checkpoint = checkpoint
        self.gcon = gcon

        if self.shared_conv:
            self.features_extractors = GraphConv(
                self.hidden_dim + self.rwpe_h,
                self.hidden_dim,
                num_heads=n_attention_heads,
                edge_scoring=self.update_edge_features,
                pyg=self.pyg,
            )
        else:
            self.features_extractors = torch.nn.ModuleList()
            self.mlps = torch.nn.ModuleList()
            self.edge_embedders = torch.nn.ModuleList()

        for layer in range(self.n_layers_features_extractor):
            if self.normalize:
                self.norms.append(torch.nn.LayerNorm(self.hidden_dim))
                if self.residual:
                    self.normsbis.append(torch.nn.LayerNorm(self.hidden_dim))

            if self.rwpe_k != 0:
                self.pe_conv.append(
                    GraphConv(
                        self.rwpe_h,
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                        bias=False,
                        pyg=self.pyg,
                    )
                )
                self.pe_mlp.append(
                    MLP(
                        n_layers=n_mlp_layers_features_extractor,
                        input_dim=self.rwpe_h * n_attention_heads,
                        hidden_dim=self.rwpe_h * n_attention_heads,
                        output_dim=self.rwpe_h,
                        norm=self.normalize,
                        activation="tanh",
                    )
                )
            if not self.shared_conv:
                self.features_extractors.append(
                    GraphConv(
                        self.hidden_dim + self.rwpe_h,
                        self.hidden_dim,
                        num_heads=n_attention_heads,
                        edge_scoring=self.update_edge_features,
                        pyg=self.pyg,
                        gcon=self.gcon,
                        n_mlp_layers=n_mlp_layers_features_extractor,
                        norm=self.normalize,
                        activation=activation_features_extractor,
                    )
                )

                self.edge_embedders.append(
                    PspEdgeEmbedder(
                        edge_embedding_flavor,
                        max_n_resources,
                        self.hidden_dim,
                        add_rp_edges,
                        factored_rp,
                        n_mlp_layers_features_extractor,
                        activation_features_extractor,
                    )
                )

            if self.update_edge_features:
                self.mlps_edges.append(
                    MLP(
                        n_layers=n_mlp_layers_features_extractor,
                        input_dim=self.hidden_dim * n_attention_heads,
                        hidden_dim=self.hidden_dim * n_attention_heads,
                        output_dim=self.hidden_dim,
                        norm=self.normalize,
                        activation=activation_features_extractor,
                    )
                )
            if self.update_edge_features_pe and rwpe_k != 0:
                self.mlps_edges_pe.append(
                    MLP(
                        n_layers=n_mlp_layers_features_extractor,
                        input_dim=self.hidden_dim * n_attention_heads,
                        hidden_dim=self.hidden_dim * n_attention_heads,
                        output_dim=self.hidden_dim,
                        norm=self.normalize,
                        activation=activation_features_extractor,
                    )
                )

    # @torch.autocast(device_type="cuda")
    def forward(self, g, features, pe):
        if self.normalize:
            features = self.norm1(features)

        if self.layer_pooling == "all":
            features_list = []
            edge_scores_list = []
            if self.rwpe_k != 0:
                features_list.append(torch.cat([features, pe], dim=-1))
            else:
                features_list.append(features)

        if self.layer_pooling == "last":
            previous_feat = features
            if self.update_edge_features:
                previous_edge_scores = edge_scores
            if self.rwpe_k != 0:
                previous_pe = pe

        for layer in range(self.n_layers_features_extractor):
            if self.rwpe_k != 0:
                features = torch.cat([features, pe], dim=-1)

            if self.update_edge_features:
                features, _, new_e_attr = self.features_extractors[layer](
                    g, features, edge_features, efeats_e=edge_scores
                )
                edge_scores = self.mlps_edges[layer](
                    new_e_attr.flatten(start_dim=-2, end_dim=-1)
                )
                features = self.mlps[layer](features.flatten)

            else:
                if self.shared_conv:
                    features = self.features_extractors(
                        g._graph,
                        features,
                        edge_features,
                    )
                else:
                    if layer % self.checkpoint:
                        features, _ = torch.utils.checkpoint.checkpoint(
                            self.features_extractors[layer],
                            g._graph,
                            features,
                            edge_features,
                            use_reentrant=False,
                        )
                        features = torch.utils.checkpoint.checkpoint(
                            self.mlps[layer], features, use_reentrant=False
                        )
                    else:
                        features, _ = self.features_extractors[layer](
                            g._graph,
                            features,
                            self.edge_embedders[layer](g),
                        )
                        features = self.mlps[layer](features)

            if self.rwpe_k != 0:
                if self.update_edge_features_pe:
                    pe, new_e_pe_attr = self.pe_conv[layer](
                        g._graph, pe, edge_features_pe.clone()
                    )
                    edge_features_pe += self.mlps_edges_pe[layer](
                        new_e_pe_attr.flatten(start_dim=-2, end_dim=-1)
                    )
                else:
                    pe, _ = self.pe_conv[layer](g._graph, pe, edge_features_pe)
                pe = self.pe_mlp[layer](pe)

            # if self.layer_pooling == "all":
            #     features_list.append(features)
            if self.normalize:
                features = self.norms[layer](features)

            if self.residual:
                if self.layer_pooling == "all":
                    features += features_list[-1][:, : self.hidden_dim]
                    if self.update_edge_features:
                        edge_scores += edge_scores_list[-1]
                    if self.rwpe_k != 0:
                        pe += features_list[-1][:, self.hidden_dim :]
                else:
                    features += previous_feat[:, : self.hidden_dim]
                    if self.update_edge_features:
                        edge_scores += previous_edge_scores
                        previous_edge_scores = edge_scores
                    if self.rwpe_k != 0:
                        pe += previous_pe
                if self.normalize:
                    features = self.normsbis[layer](features)
                if self.layer_pooling == "last":
                    previous_feat = features
                    if self.rwpe_k != 0:
                        previous_pe = pe

            if self.layer_pooling == "all":
                if self.rwpe_k != 0:
                    features_list.append(torch.cat([features, pe], dim=-1))
                else:
                    features_list.append(features)
                if self.update_edge_features:
                    edge_scores_list.append(edge_scores)

        if self.layer_pooling == "all":
            return features_list
        return features
