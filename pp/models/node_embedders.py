import re
import torch
from generic.mlp import MLP


class SimpleNodeEmbedder(torch.nn.Module):
    def __init__(
        self,
        output_dim,
        n_layers,
        activation,
        lappe=None,
        rwpe=None,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.nfeats = 8
        self.lappe = lappe if lappe is not None else 0
        self.rwpe = rwpe if rwpe is not None else 0
        self.mlp = MLP(
            n_layers,
            self.nfeats + self.lappe + self.rwpe,
            output_dim,
            output_dim,
            False,
            activation,
        )

    def forward(self, g, nid):
        x = torch.empty(
            (nid.shape[0], self.nfeats + self.lappe + self.rwpe),
            device=self.mlp.layers[0].weight.device,
        )
        x[:, 0] = self._get_agent_feature(g, nid, "cur_pos")
        x[:, 1] = g.start[nid]
        x[:, 2] = g.goal[nid]
        x[:, 3] = g.degree[nid]
        x[:, 4] = self._get_agent_feature(g, nid, "n_breaks")
        danger = getattr(g, "danger", None)
        if danger is not None:
            x[:, 5] = danger[nid].to(x.device).float()
        else:
            x[:, 5] = 0.0
        x[:, 6 : self.nfeats] = g.norm_coord[nid]
        if self.lappe != 0:
            x[:, self.nfeats : self.nfeats + self.lappe] = g.laplacian_eigenvector_pe[
                nid
            ]
        if self.rwpe != 0:
            x[:, self.nfeats + self.lappe : self.nfeats + self.lappe + self.rwpe] = (
                g.random_walk_pe[nid]
            )
        return self.mlp(x)

    def _get_agent_feature(self, graph, nid, feature_name):
        if hasattr(graph, feature_name):
            return getattr(graph, feature_name)[nid].float()
        prefixed_name = f"agent_0_{feature_name}"
        if hasattr(graph, prefixed_name):
            return getattr(graph, prefixed_name)[nid].float()
        print("failed to find feature", feature_name)
        return torch.zeros_like(getattr(graph, "start")[nid]).float()


class SimpleNodeEmbedderNonChrono(torch.nn.Module):
    def __init__(
        self,
        output_dim,
        n_layers,
        activation,
        nonchrono,
        lappe=None,
        rwpe=None,
        max_order=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.max_order = max_order
        # NPBB
        if max_order is not None:
            self.nfeats = max_order + 6
        else:
            self.nfeats = 8
        # self.nfeats = 7
        self.lappe = lappe if lappe is not None else 0
        self.rwpe = rwpe if rwpe is not None else 0
        self.nonchrono = nonchrono
        self.mlp = MLP(
            n_layers,
            self.nfeats + self.lappe + self.rwpe,
            output_dim,
            output_dim,
            False,
            activation,
        )

    def forward(self, g, nid):
        x = torch.empty(
            (nid.shape[0], self.nfeats + self.lappe + self.rwpe),
            device=self.mlp.layers[0].weight.device,
        )
        x[:, 0] = g.start[nid]
        x[:, 1] = g.goal[nid]
        # NPB below
        if self.nonchrono == "path":
            x[:, 2] = g.in_path[nid]
        else:
            x[:, 2] = 0
        x[:, 3] = g.degree[nid]
        x[:, 4] = self._get_agent_feature(g, nid, "selected")
        danger = getattr(g, "danger", None)
        if danger is not None:
            x[:, 5] = danger[nid].to(x.device).float()
        else:
            x[:, 5] = 0.0
        x[:, 6 : self.nfeats] = g.norm_coord[nid]

        # vanilla below
        # x[:, 2] = g.degree[nid]
        # x[:, 3] = self._get_agent_feature(g, nid, "selected")
        # danger = getattr(g, "danger", None)
        # if danger is not None:
        #     x[:, 4] = danger[nid].to(x.device).float()
        # else:
        #     x[:, 4] = 0.0
        # x[:, 5 : self.nfeats] = g.norm_coord[nid]
        if self.lappe != 0:
            x[:, self.nfeats : self.nfeats + self.lappe] = g.laplacian_eigenvector_pe[
                nid
            ]
        if self.rwpe != 0:
            x[:, self.nfeats + self.lappe : self.nfeats + self.lappe + self.rwpe] = (
                g.random_walk_pe[nid]
            )

        return self.mlp(x)

    def _get_agent_feature(self, graph, nid, feature_name):
        if hasattr(graph, feature_name):
            return getattr(graph, feature_name)[nid].float()
        prefixed_name = f"agent_0_{feature_name}"
        if hasattr(graph, prefixed_name):
            return getattr(graph, prefixed_name)[nid].float()
        return torch.zeros_like(getattr(graph, "start")[nid]).float()


class PoolNodeEmbedder(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # TODO : should be only a param vector
        self.emb = torch.nn.Embedding(1, output_dim)

    def forward(self, g, nid):
        return self.emb(torch.tensor([0] * len(nid), device=self.emb.weight.device))


class MASimpleNodeEmbedder(torch.nn.Module):
    def __init__(
        self,
        output_dim,
        n_layers,
        activation,
        lappe=None,
        rwpe=None,
        danger=False,
        breaks=False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.activation = activation
        self.lappe = lappe if lappe is not None else 0
        self.rwpe = rwpe if rwpe is not None else 0
        self.agent_feature_order = (
            "start",
            "cur_pos",
            "visited",
            "selected",
            "n_breaks",
        )
        self.mlp = None
        self._mlp_input_dim = None
        self._agent_key_pattern = re.compile(r"agent_(\d+)_(.+)")

    def forward(self, g, nid):
        features = []

        start_feat = getattr(g, "start")[nid].unsqueeze(-1).float()
        features.append(start_feat)
        goal_feat = getattr(g, "goal")[nid].unsqueeze(-1).float()
        features.append(goal_feat)
        degree_feat = getattr(g, "degree")[nid].unsqueeze(-1).float()
        features.append(degree_feat)
        norm_coord = getattr(g, "norm_coord")[nid].float()
        features.append(norm_coord)
        if hasattr(g, "danger"):
            features.append(getattr(g, "danger")[nid].unsqueeze(-1).float())
        else:
            features.append(torch.zeros_like(start_feat))

        agent_ids = self._resolve_agent_ids(g)
        base_device = start_feat.device
        num_nodes = nid.shape[0]

        # print('embedding multi-agent values')
        for agent_id in agent_ids:
            prefix = f"agent_{agent_id}_"
            for feature_name in self.agent_feature_order:
                key = prefix + feature_name
                if hasattr(g, key):
                    value = getattr(g, key)[nid].unsqueeze(-1).float()
                else:
                    value = torch.zeros((num_nodes, 1), device=base_device)
                features.append(value)
                # print('added feature', key, ' / value=', value.shape)

        x = torch.cat(features, dim=1)

        if self.lappe:
            x = torch.cat([x, getattr(g, "laplacian_eigenvector_pe")[nid]], dim=1)
        if self.rwpe:
            x = torch.cat([x, getattr(g, "random_walk_pe")[nid]], dim=1)

        if self.mlp is None or self._mlp_input_dim != x.size(1):
            self._initialize_mlp(x.size(1), x.device)
        elif next(self.mlp.parameters()).device != x.device:
            self.mlp = self.mlp.to(x.device)

        return self.mlp(x)

    def _initialize_mlp(self, input_dim, device):
        mlp = MLP(
            self.n_layers,
            input_dim,
            self.output_dim,
            self.output_dim,
            False,
            self.activation,
        )
        self._mlp_input_dim = input_dim
        # if hasattr(torch, "compile"):
        #     mlp = torch.compile(mlp, dynamic=True)
        self.mlp = mlp.to(device)

    def _resolve_agent_ids(self, graph):
        agent_ids = set()
        keys = graph.keys() if callable(getattr(graph, "keys", None)) else graph.keys
        for key in keys:
            match = self._agent_key_pattern.match(key)
            if match:
                agent_ids.add(int(match.group(1)))
        return sorted(agent_ids)
