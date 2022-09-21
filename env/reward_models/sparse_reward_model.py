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
        is_done = (features_tp[:, 0] == 1).all().item()
        if not is_done:
            return 0
        makespan = torch.max(features_tp[:, 1]).item()
        # We don't want |reward| to be > 1. Since makespan is divided by longest task, we only have to divide by 2
        reward = makespan / 2
        return -reward
