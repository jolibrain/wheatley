import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import MLP


class MLPExtractor(nn.Module):
    def __init__(
        self,
        add_boolean,
        mlp_act,
        device,
        input_dim_features_extractor,
        max_n_nodes,
        max_n_jobs,
        n_layers_features_extractor,
        hidden_dim_features_extractor,
        n_mlp_layers_actor,
        hidden_dim_actor,
        n_mlp_layers_critic,
        hidden_dim_critic,
    ):
        super(MLPExtractor, self).__init__()

        self.max_n_nodes = max_n_nodes
        self.max_n_jobs = max_n_jobs

        # This is necessary because of stable_baselines3 API
        self.latent_dim_pi = self.max_n_jobs * (2 if add_boolean else 1)
        self.latent_dim_vf = 1
        self.add_boolean = add_boolean
        self.device = device
        self.input_dim_features_extractor = input_dim_features_extractor

        self.actor = MLP(
            n_layers=n_mlp_layers_actor,
            input_dim=(self.input_dim_features_extractor + hidden_dim_features_extractor * n_layers_features_extractor) * 2,
            hidden_dim=hidden_dim_actor,
            output_dim=2 if self.add_boolean else 1,
            batch_norm=False,
            activation=mlp_act,
            device=device,
        )
        self.critic = MLP(
            n_layers=n_mlp_layers_critic,
            input_dim=hidden_dim_features_extractor * n_layers_features_extractor + self.input_dim_features_extractor,
            hidden_dim=hidden_dim_critic,
            output_dim=1,
            batch_norm=False,
            activation=mlp_act,
            device=device,
        )

    def forward(self, graph_and_nodes_embedding):
        """
        Takes nodes_and_graph_embedding as input. This should be the output of the
        FeatureExtractor
        """

        # First decompose the features into graph_embedding and nodes_embedding
        graph_embedding, nodes_embedding = self._decompose_features(graph_and_nodes_embedding)
        batch_size = graph_embedding.shape[0]

        # Then compute actor and critic
        value = self.critic(graph_embedding)

        # Compute the pairs
        pairs_to_compute, indexes = self._get_pairs_to_compute(graph_embedding, nodes_embedding)

        # Compute the probabilities
        probabilities = self.actor(pairs_to_compute)
        if self.add_boolean:
            probabilities, boolean = torch.split(probabilities, [1, 1], dim=2)
            boolean = torch.sigmoid(boolean)
        # pi = F.softmax(probabilities, dim=1)
        # logits are expected at output, not probas
        pi = probabilities
        if self.add_boolean:
            pi_without_boolean = pi * (1 - boolean)
            pi = pi * boolean

        # And reshape pi in ordrer to have every value corresponding to its edge index
        shaped_pi = torch.zeros((batch_size, self.max_n_nodes), device=self.device)
        if self.add_boolean:
            shaped_pi_without_boolean = torch.zeros((batch_size, self.max_n_nodes), device=self.device)
        for i in range(batch_size):
            shaped_pi[i][indexes[i]] = pi[i].reshape(pi.shape[1])[0 : len(indexes[i])]
            if self.add_boolean:
                shaped_pi_without_boolean[i][indexes[i]] = pi_without_boolean[i].reshape(pi_without_boolean.shape[1])[
                    0 : len(indexes[i])
                ]

        # The final pi must include all actions (even those that are not applicable
        # because of the size of the graph). We complete the vector to get a vector
        # self.max_n_jobs with 0.
        filled_pi = torch.zeros((batch_size, self.max_n_nodes * (2 if self.add_boolean else 1)), device=self.device)
        filled_pi[:, 0 : shaped_pi.shape[1]] = shaped_pi
        if self.add_boolean:
            filled_pi[
                :, self.max_n_nodes : self.max_n_nodes + shaped_pi_without_boolean.shape[1]
            ] = shaped_pi_without_boolean
        return filled_pi, value

    def _get_pairs_to_compute(self, g_embedding, n_embeddings):
        batch_size = n_embeddings.shape[0]
        n_nodes = n_embeddings.shape[1]
        mask = torch.ones(batch_size, n_nodes, 1)
        indexes = []
        masked_tensors = []

        dim = int(torch.max(torch.sum(mask, axis=1)).item())
        for i in range(mask.shape[0]):
            # Every mask should be the same size, so if there are multiple sizes, take the largest one
            indexes.append((mask[i] == 1).nonzero(as_tuple=True)[0])
            masked_tensor = torch.zeros((dim, g_embedding.shape[2] + 1 * n_embeddings.shape[2]), device=self.device)
            counter = 0
            for index in indexes[-1]:
                node = n_embeddings[i][index]
                masked_tensor[counter : counter + 1, :] = torch.cat([g_embedding[i][0], node])
                counter += 1
            masked_tensors.append(masked_tensor)
        return torch.stack(masked_tensors), indexes

    def _decompose_features(self, graph_and_nodes_embedding):
        graph_embedding, nodes_embedding = torch.split(
            graph_and_nodes_embedding,
            [1, graph_and_nodes_embedding.shape[1] - 1],
            dim=1,
        )
        return graph_embedding, nodes_embedding
