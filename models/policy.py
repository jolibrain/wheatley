from functools import partial
import numpy as np
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from torch import nn
import torch


class Policy(MaskableActorCriticPolicy):
    def extract_features(self, obs):
        # skip preprocessing
        return self.features_extractor(obs)

    def _build(self, lr_schedule):
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, 1)
        self.full_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.reduce_value_net = nn.Linear(3, 1)

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
                self.action_net: 1,
                self.full_value_net: 1,
                self.reduce_value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        fe_lr = self.optimizer_kwargs["fe_lr"] if self.optimizer_kwargs["fe_lr"] is not None else self.optimizer_kwargs["lr"]
        pgroup = [
            {"params": self.features_extractor.parameters(), "lr": fe_lr},
        ]

        self.optimizer = self.optimizer_class(pgroup, lr=lr_schedule(1))
        print("optimizer", self.optimizer)

    def value_net(self, latent_pi):
        values = self.full_value_net(latent_pi)
        mini = values.min(dim=1)[0]
        maxi = values.max(dim=1)[0]
        mean = values.mean(dim=1)
        mini_maxi_mean = torch.cat((mini, maxi, mean), dim=1)
        value = self.reduce_value_net(mini_maxi_mean)
        return value
