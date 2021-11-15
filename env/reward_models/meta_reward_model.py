from env.reward_model import RewardModel


class MetaRewardModel(RewardModel):
    def __init__(self, reward_model_cls_list, reward_model_kwargs_list, coefs, n_timesteps):
        self.reward_models = [
            reward_model_cls_list[i](**reward_model_kwargs_list[i]) for i in range(len(reward_model_cls_list))
        ]
        self.coefs = coefs
        self.cur_coefs = coefs
        self.n_timesteps = n_timesteps

    def evaluate(self, obs, action, next_obs):
        rewards = [self.reward_models[i].evaluate(obs, action, next_obs) for i in range(len(self.reward_models))]
        reward = 0
        for i in range(len(self.reward_models)):
            reward += self.cur_coefs[i] * rewards[i]
        self.cur_coefs[0] = min(1, self.cur_coefs[0] + (1 - self.coefs[0]) / self.n_timesteps)
        self.cur_coefs[1] = max(0, self.cur_coefs[1] - self.coefs[1] / self.n_timesteps)
        return reward
