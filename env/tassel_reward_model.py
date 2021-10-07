import torch

from env.reward_model import RewardModel
from utils.utils import node_to_job_and_task


class TasselRewardModel(RewardModel):
    def __init__(self, affectations, durations):
        self.affectations = affectations
        self.durations = durations

    def evaluate(self, obs, action, next_obs):
        """
        See https://arxiv.org/pdf/2104.03760.pdf for reward implementation explaination
        """
        features_t = obs.features
        features_tp = next_obs.features
        scheduled_node_id = (features_tp[:, 0] - features_t[:, 0]).nonzero(as_tuple=True)[0].item()
        job_id, task_id = node_to_job_and_task(scheduled_node_id, self.affectations.shape[1])
        cur_duration = self.durations[job_id, task_id]

        ancient_idle_time = torch.sum(torch.max(features_t[:, 1].reshape(self.affectations.shape), axis=1).values)
        print(ancient_idle_time)
        new_idle_time = torch.sum(torch.max(features_tp[:, 1].reshape(self.affectations.shape), axis=1).values)
        print(new_idle_time)
        print(cur_duration)
        reward = cur_duration - (new_idle_time - ancient_idle_time)
        return reward.item()
