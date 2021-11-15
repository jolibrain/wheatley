import torch

from env.reward_model import RewardModel


class L2DRewardModel(RewardModel):
    def __init__(self):
        pass

    def evaluate(self, obs, action, next_obs):
        """
        Reward is computed as H(s_t) - H(s_t+1) where H(s_t) = max{tasks_lower_bound} at
        time s_t. For more info, see https://arxiv.org/abs/2010.12367
        """
        features_t = obs.features
        features_tp = next_obs.features
        H_st = torch.max(features_t[:, 7]).item()
        H_stp = torch.max(features_tp[:, 7]).item()
        reward = H_st - H_stp
        return reward
