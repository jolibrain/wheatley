from functools import partial
import numpy as np
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
import torch
from torch.distributions import Categorical

from models.mlp_extractor import MLPExtractor
from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution


class Policy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        add_boolean = kwargs.pop("add_boolean")
        self.add_boolean = add_boolean
        mlp_act = kwargs.pop("mlp_act")
        self.mlp_act = mlp_act
        _device = kwargs.pop("_device")  # We can't use device since it's already used. So we use _device
        self._device = _device
        input_dim_features_extractor = kwargs.pop("input_dim_features_extractor")
        self.input_dim_features_extractor = input_dim_features_extractor
        self.ortho_init = True
        max_n_nodes = kwargs.pop("max_n_nodes")
        self.max_n_nodes = max_n_nodes
        max_n_jobs = kwargs.pop("max_n_jobs")
        self.max_n_jobs = max_n_jobs
        n_layers_features_extractor = kwargs.pop("n_layers_features_extractor")
        self.n_layers_features_extractor = n_layers_features_extractor
        hidden_dim_features_extractor = kwargs.pop("hidden_dim_features_extractor")
        self.hidden_dim_features_extractor = hidden_dim_features_extractor
        n_mlp_layers_actor = kwargs.pop("n_mlp_layers_actor")
        self.n_mlp_layers_actor = n_mlp_layers_actor
        hidden_dim_actor = kwargs.pop("hidden_dim_actor")
        self.hidden_dim_actor = hidden_dim_actor
        n_mlp_layers_critic = kwargs.pop("n_mlp_layers_critic")
        self.n_mlp_layers_critic = n_mlp_layers_critic
        hidden_dim_critic = kwargs.pop("hidden_dim_critic")
        self.hidden_dim_critic = hidden_dim_critic
        super(Policy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = MLPExtractor(
            self.add_boolean,
            self.mlp_act,
            self._device,
            self.input_dim_features_extractor,
            self.max_n_nodes,
            self.max_n_jobs,
            self.n_layers_features_extractor,
            self.hidden_dim_features_extractor,
            self.n_mlp_layers_actor,
            self.hidden_dim_actor,
            self.n_mlp_layers_critic,
            self.hidden_dim_critic,
        )

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
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_latent(self, obs):
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        return latent_pi, latent_vf
        
    def forward(self, obs, deterministic=False, action_masks=None):
        """
        Forward pass in all the networks (actor and critic)
        Note : this is a reimplementation of the forward function of
        stable_baselines3.common.polcies.ActorCriticPolicy, in order to compute actor
        and critic net in a certain way.
        """
        latent_pi, latent_vf = self._get_latent(obs)
        values = latent_vf  # Modification here
        distribution = self.action_dist.proba_distribution(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False, action_masks=None):
        actions, _, _ = self.forward(observation, deterministic, action_masks)
        return actions

    def predict_values(self, observation, deterministic=False, action_masks=None):
        _, values, _ = self.forward(observation, deterministic, action_masks)
        return values

    def evaluate_actions(self, obs, actions, action_masks=None):
        latent_pi, latent_vf = self._get_latent(obs)
        distribution = self.action_dist.proba_distribution(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = latent_vf
        return values, log_prob, distribution.entropy()
