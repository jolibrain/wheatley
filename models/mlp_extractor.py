import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import MLP

from config import (
    HIDDEN_DIM_FEATURES_EXTRACTOR,
    N_MLP_LAYERS_ACTOR,
    HIDDEN_DIM_ACTOR,
    N_MLP_LAYERS_CRITIC,
    HIDDEN_DIM_CRITIC,
    MAX_N_NODES,
    DEVICE,
)


class MLPExtractor(nn.Module):
    def __init__(self, add_pdr_boolean):
        super(MLPExtractor, self).__init__()

        # This is necessary because of stable_baselines3 API
        self.latent_dim_pi = (MAX_N_NODES ** 2) * (2 if add_pdr_boolean else 1)
        self.latent_dim_vf = 1
        self.add_pdr_boolean = add_pdr_boolean

        self.actor = MLP(
            n_layers=N_MLP_LAYERS_ACTOR,
            input_dim=HIDDEN_DIM_FEATURES_EXTRACTOR * 3,
            hidden_dim=HIDDEN_DIM_ACTOR,
            output_dim=2 if self.add_pdr_boolean else 1,
            batch_norm=False,
            activation="tanh",
            device=DEVICE,
        )
        self.critic = MLP(
            n_layers=N_MLP_LAYERS_CRITIC,
            input_dim=HIDDEN_DIM_FEATURES_EXTRACTOR,
            hidden_dim=HIDDEN_DIM_CRITIC,
            output_dim=1,
            batch_norm=False,
            activation="tanh",
            device=DEVICE,
        )

    def forward(self, embedded_features):
        """
        Takes nodes_and_graph_embedding as input. This should be the output of the
        FeatureExtractor
        """

        # First decompose the features into mask, graph_embedding and nodes_embedding
        graph_embedding, nodes_embedding, mask = self._decompose_features(embedded_features)
        n_nodes = nodes_embedding.shape[1]
        batch_size = graph_embedding.shape[0]

        # Then compute actor and critic
        value = self.critic(graph_embedding)

        # Compute the pairs and apply the mask
        pairs_to_compute, indexes = self._get_pairs_to_compute(graph_embedding, nodes_embedding, mask)

        # Compute the probabilities
        probabilities = self.actor(pairs_to_compute)
        if self.add_pdr_boolean:
            probabilities, pdr_boolean = torch.split(probabilities, [1, 1], dim=2)
            pdr_boolean = torch.sigmoid(pdr_boolean)
        pi = F.softmax(probabilities, dim=1)
        if self.add_pdr_boolean:
            pi_without_pdr = pi * (1 - pdr_boolean)
            pi = pi * pdr_boolean

        # And reshape pi in ordrer to have every value corresponding to its edge index
        shaped_pi = torch.zeros((batch_size, n_nodes * n_nodes), device=DEVICE)
        if self.add_pdr_boolean:
            shaped_pi_without_pdr = torch.zeros((batch_size, n_nodes * n_nodes), device=DEVICE)
        for i in range(batch_size):
            shaped_pi[i][indexes[i]] = pi[i].reshape(pi.shape[1])[0 : len(indexes[i])]
            if self.add_pdr_boolean:
                shaped_pi_without_pdr[i][indexes[i]] = pi_without_pdr[i].reshape(pi_without_pdr.shape[1])[
                    0 : len(indexes[i])
                ]

        # The final pi must include all actions (even those that are not applicable
        # because of the size of the graph). So we convert the flattened pi to a square
        # matrix of size (n_nodes, n_nodes). We then complete the matrix to get a square
        # matrix of size (MAX_N_NODES, MAX_N_NODES) with 0, and we reflatten it.
        shaped_pi = shaped_pi.reshape(batch_size, n_nodes, n_nodes)
        if self.add_pdr_boolean:
            shaped_pi_without_pdr = shaped_pi_without_pdr.reshape(batch_size, n_nodes, n_nodes)
        filled_pi = torch.zeros((batch_size, MAX_N_NODES * (2 if self.add_pdr_boolean else 1), MAX_N_NODES), device=DEVICE)
        filled_pi[:, 0:n_nodes, 0:n_nodes] = shaped_pi
        if self.add_pdr_boolean:
            filled_pi[:, MAX_N_NODES : MAX_N_NODES + n_nodes, 0:n_nodes] = shaped_pi_without_pdr
        filled_pi = filled_pi.reshape(batch_size, MAX_N_NODES * MAX_N_NODES * (2 if self.add_pdr_boolean else 1))

        return filled_pi, value

    def _get_pairs_to_compute(self, g_embedding, n_embeddings, mask):
        indexes = []
        masked_tensors = []

        n_nodes = n_embeddings.shape[1]

        dim = int(torch.max(torch.sum(mask, axis=1)).item())
        for i in range(mask.shape[0]):
            # Every mask should be the same size, so if there are multiple sizes, take the largest one
            indexes.append((mask[i] == 1).nonzero(as_tuple=True)[0])
            masked_tensor = torch.zeros((dim, g_embedding.shape[2] + 2 * n_embeddings.shape[2]), device=DEVICE)

            counter = 0
            for index in indexes[-1]:
                index1 = torch.div(index, n_nodes, rounding_mode="floor")
                index2 = index % n_nodes
                node1 = n_embeddings[i][index1]
                node2 = n_embeddings[i][index2]
                masked_tensor[counter : counter + 1, :] = torch.cat([g_embedding[i][0], node1, node2])
                counter += 1
            masked_tensors.append(masked_tensor)
        return torch.stack(masked_tensors), indexes

    def _decompose_features(self, embedded_features):
        graph_and_nodes_embedding, extended_mask = torch.split(
            embedded_features,
            [
                HIDDEN_DIM_FEATURES_EXTRACTOR,
                embedded_features.shape[2] - HIDDEN_DIM_FEATURES_EXTRACTOR,
            ],
            dim=2,
        )
        graph_embedding, nodes_embedding = torch.split(
            graph_and_nodes_embedding,
            [1, graph_and_nodes_embedding.shape[1] - 1],
            dim=1,
        )
        batch_size = graph_embedding.shape[0]
        mask = extended_mask[:, 1:, :].reshape(batch_size, -1)
        return graph_embedding, nodes_embedding, mask
