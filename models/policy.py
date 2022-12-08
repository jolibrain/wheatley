from functools import partial
import numpy as np
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from torch import nn
import torch
from .mlp import MLP


class GraphExtractor(nn.Module):
    def __init__(self):
        super(GraphExtractor, self).__init__()

    def forward(self, features):
        return features, features[:, 0, features.shape[2] // 2 :]

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        # filter out node specific features
        return features[:, 0, features.shape[2] // 2 :]


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
        self.mlp_extractor = GraphExtractor()

        self.action_net = MLP(
            len(self.net_arch[0]["pi"]),
            self.features_dim,
            self.net_arch[0]["pi"][0],
            1,
            False,
            self.activation_fn,
            self.device,
        )
        self.value_net = MLP(
            len(self.net_arch[0]["vf"]),
            self.features_dim // 2,
            self.net_arch[0]["vf"][0],
            1,
            False,
            self.activation_fn,
            self.device,
        )

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                # self.features_extractor: np.sqrt(2),
                self.features_extractor: 1,
                self.action_net: 1,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        fe_lr = self.optimizer_kwargs["fe_lr"] if self.optimizer_kwargs["fe_lr"] is not None else self.optimizer_kwargs["lr"]
        pgroup = [
            {"params": self.features_extractor.parameters(), "lr": fe_lr},
        ]

        self.optimizer = self.optimizer_class(pgroup, lr=lr_schedule(1))
        print("optimizer", self.optimizer)
