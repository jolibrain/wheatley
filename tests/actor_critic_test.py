import sys

sys.path.append("..")

import torch
from torch_geometric.data import Data, DataLoader

from models.actor_critic import ActorCritic

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = ActorCritic(
        n_jobs=2,
        n_machines=2,
        n_mlp_layers_feature_extractor=3,
        n_layers_feature_extractor=3,
        input_dim_feature_extractor=2,
        hidden_dim_feature_extractor=3,
        n_mlp_layers_actor=3,
        hidden_dim_actor=64,
        n_mlp_layers_critic=3,
        hidden_dim_critic=64,
        device=device,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 2, 2, 2, 3], [0, 3, 0, 1, 2, 3, 0, 2, 3, 3]],
        dtype=torch.long,
    )
    features = torch.rand(4, 2)
    graph = Data(x=features, edge_index=edge_index)
    dataloader = DataLoader([graph], batch_size=1)
    for batch in dataloader:
        print(actor_critic(batch))
    print(actor_critic(graph=graph))
