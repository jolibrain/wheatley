from functools import partial
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from torch.distributions import Categorical

from models.mlp_extractor import MLPExtractor


class Policy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = MLPExtractor()

    # The four following functions should be checked, since they are re written from
    # stable_baselines3. We should also check that there are no other functions that
    # are used by PPO, to guarantee that it will work

    def _build(self, lr_schedule):
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

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
        actions = (
            torch.argmax(distribution.probs, dim=1)
            if deterministic
            else distribution.sample()
        )
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False):
        actions, _, _ = self.forward(observation, deterministic)
        return actions

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf, _ = self._get_latent(obs)
        distribution = Categorical(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = latent_vf
        return values, log_prob, distribution.entropy()
