import numpy as np
import torch

from env.reward_model import RewardModel
from utils.utils import node_to_job_and_task


class TasselRewardModel(RewardModel):
    def __init__(self, affectations, durations, normalize_input):
        self.affectations = affectations
        self.durations = durations
        self.dividing_factor = np.sum(self.durations.flatten()) if normalize_input else 1

    def evaluate(self, obs, action, next_obs):
        """
        See https://arxiv.org/pdf/2104.03760.pdf for reward implementation explaination
        """
        features_t = obs.features
        features_tp = next_obs.features
        scheduled_node_id = (features_tp[:, 0] - features_t[:, 0]).nonzero(as_tuple=True)[0].item()
        job_id, task_id = node_to_job_and_task(scheduled_node_id, self.affectations.shape[1])
        cur_duration = self.durations[job_id, task_id]

        ancient_idle_time = torch.sum(torch.max(features_t[:, 7].reshape(self.affectations.shape), axis=1).values)
        new_idle_time = torch.sum(torch.max(features_tp[:, 7].reshape(self.affectations.shape), axis=1).values)
        reward = (cur_duration / self.dividing_factor) - (new_idle_time - ancient_idle_time)
        return reward.item()
