from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from torch.distributions import Categorical

from models.mlp_extractor import MLPExtractor


class Policy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = MLPExtractor()

    # The three following functions should be checked, since they are re written from
    # stable_baselines3. We should also check that there are no other functions that
    # are used by PPO, to guarantee that it will work

    def forward(self, obs, deterministic=False):
        """
        Forward pass in all the networks (actor and critic)
        Note : this is a reimplementation of the forward function of
        stable_baselines3.common.polcies.ActorCriticPolicy, in order to compute actor
        and critic net in a certain way.
        """
        latent_pi, latent_vf, _ = self._get_latent(obs)
        values = latent_vf  # Modification here
        distribution = Categorical(latent_pi)  # And here
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False):
        actions, _, _ = self.forward(observation, deterministic)
        return actions

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf, _ = self._get_latent(obs)
        distribution = Categorical(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
