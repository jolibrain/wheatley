from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

from models.mlp_extractor import MLPExtractor


class Policy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = MLPExtractor()

    def forward(self, obs, deterministic=False):
        """Forward pass in all the networks (actor and critic)
        Note : this is a reimplementation of the forward function of
        stable_baselines3.common.polcies.ActorCriticPolicy, in order to compute actor
        and critic net in a certain way.
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observation
        values = latent_vf  # Modification here
        distribution = self.action_dist.proba_distribution(
            action_logits=latent_pi
        )
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
