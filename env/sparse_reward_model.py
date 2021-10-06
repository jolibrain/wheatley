import torch

from env.reward_model import RewardModel


class SparseRewardModel(RewardModel):
    def __init__(self):
        pass

    def evaluate(self, obs, action, next_obs):
        """
        Reward is 0 for every time steps, except for the last one, where it is the opposite of the Makespan
        """
        features_tp = next_obs.features
        is_done = (features_tp[:, 0] == 0).all().item()
        makespan = torch.max(features_tp[:, 1]).item()
        return -makespan if is_done else 0
