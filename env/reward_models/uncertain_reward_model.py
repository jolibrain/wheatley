import torch

from env.reward_model import RewardModel


class UncertainRewardModel(RewardModel):
    def __init__(self, config):
        # 4 uses real completion time
        # 5 is for min
        # 6 is for max
        # 7 is for mode
        if config == "optimistic":
            self.index = 5
        elif config == "pessimistic":
            self.index = 6
        elif config == "realistic":
            self.index = 4
        elif config == "averagistic":
            self.index = 7
        else:
            raise Exception("Reward model not recognized :  " + config)

    def evaluate(self, obs, action, next_obs):
        """
        Reward is computed as H(s_t) - H(s_t+1) where H(s_t) = max{tasks_lower_bound} at
        time s_t. For more info, see https://arxiv.org/abs/2010.12367
        """
        features_t = obs.features
        features_tp = next_obs.features
        H_st = torch.max(features_t[:, self.index]).item()
        H_stp = torch.max(features_tp[:, self.index]).item()
        reward = H_st - H_stp
        return reward
