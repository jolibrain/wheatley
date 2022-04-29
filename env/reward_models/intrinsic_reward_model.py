import torch
import torch.nn as nn

from env.reward_model import RewardModel


class IntrinsicRewardModel(RewardModel):
    def __init__(self, observation_input_size, n_nodes):
        self.random_network = nn.Sequential(
            nn.Linear(observation_input_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16), nn.Sigmoid()
        )
        self.predictor_network = nn.Sequential(
            nn.Linear(observation_input_size, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.Sigmoid()
        )
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=0.01)
        self.n_nodes = n_nodes

    def evaluate(self, obs, action, next_obs):
        inp = obs.features.flatten()
        self.optimizer.zero_grad()
        output = self.predictor_network(inp)
        target = self.random_network(inp)
        target = torch.clip(target, -0.5, 0.5)
        loss = self.criterion(output, target)
        reward = loss.item()
        loss.backward()
        self.optimizer.step()
        reward = reward / self.n_nodes  # We divide by the number of nodes to have a return which is between -1 and 1
        return reward
