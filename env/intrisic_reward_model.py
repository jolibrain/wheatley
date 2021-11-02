import torch
import torch.nn as nn

from env.reward_model import RewardModel


class IntrisicRewardModel(RewardModel):
    def __init__(self, observation_input_size):
        self.random_network = nn.Sequential(
            nn.Linear(observation_input_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16), nn.Sigmoid()
        )
        self.predictor_network = nn.Sequential(
            nn.Linear(observation_input_size, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.Sigmoid()
        )
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=0.001)

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
        return reward
