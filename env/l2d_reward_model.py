import torch


class L2DRewardModel:
    def __init__(self):
        pass

    def evaluate(self, state, action, next_state):
        """
        Reward is computed as H(s_t) - H(s_t+1) where H(s_t) = max{tasks_lower_bound} at
        time s_t. For more info, see https://arxiv.org/abs/2010.12367
        """
        features_t = state.features
        features_tp = next_state.features
        H_st = torch.max(features_t[:, 1:]).item()
        H_stp = torch.max(features_tp[:, 1:]).item()
        reward = H_st - H_stp
        return reward
