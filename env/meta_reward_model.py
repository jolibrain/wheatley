from env.reward_model import RewardModel


class MetaRewardModel(RewardModel):
    def __init__(self, reward_model_cls_list, reward_model_kwargs_list, coefs):
        self.reward_models = [
            reward_model_cls_list[i](**reward_model_kwargs_list[i]) for i in range(len(reward_model_cls_list))
        ]
        self.coefs = coefs

    def evaluate(self, obs, action, next_obs):
        rewards = [self.reward_models[i].evaluate(obs, action, next_obs) for i in range(len(self.reward_models))]
        reward = 0
        for i in range(len(self.reward_models)):
            reward += self.coefs[i] * rewards[i]
        return reward
