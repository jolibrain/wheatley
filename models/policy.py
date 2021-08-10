from stable_baselines3.common.policies import ActorCriticPolicy

from models.actor_critic import ActorCritic


class Policy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(Policy, self).__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )

    def _build_mlp_extractor(self):
        self.mlp_extractor = ActorCritic()
