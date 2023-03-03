#
# Wheatley
# Copyright (c) 2023 Jolibrain
# Authors:
#    Guillaume Infantes <guillaume.infantes@jolibrain.com>
#    Antoine Jacquet <antoine.jacquet@jolibrain.com>
#    Michel Thomazo <thomazo.michel@gmail.com>
#    Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
#
#
# This file is part of Wheatley.
#
# Wheatley is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Wheatley is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wheatley. If not, see <https://www.gnu.org/licenses/>.
#

from functools import partial
import numpy as np
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from torch import nn
import torch
from torch.distributions.utils import probs_to_logits, logits_to_probs
from .mlp import MLP
from .dadapt_adam import DAdaptAdam


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

        if self.optimizer_class == DAdaptAdam:
            self.optimizer = self.optimizer_class(self.parameters())
        else:
            if self.optimizer_kwargs["fe_lr"] is not None:
                fe_lr = self.optimizer_kwargs["fe_lr"]
                lr = self.optimizer_kwargs["lr"]
                pgroup = [
                    {"params": self.features_extractor.parameters(), "lr": fe_lr},
                    {"params": self.action_net.parameters(), "lr": lr},
                    {"params": self.value_net.parameters(), "lr": lr},
                ]
                self.optimizer = self.optimizer_class(pgroup, lr=lr_schedule(1))
            else:
                self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1))
        print("optimizer", self.optimizer)


class RPOPolicy(Policy):
    def set_rpo_smoothing_param(self, sp):
        self.rpo_smoothing_param = sp

    def _get_action_dist_from_latent(self, latent_pi, noise=False):
        action_logits = self.action_net(latent_pi)
        if noise:
            action_probs = logits_to_probs(action_logits.view(-1, action_logits.shape[-2]))
            ru = torch.rand((action_logits.shape[0]), device=action_logits.device)
            try:
                ru *= self.rpo_smoothing_param
            except AttributeError:
                # use 1 as a default
                pass
            ru = ru.unsqueeze(-1)
            action_probs = (action_probs + ru) / (1 + ru * action_logits.shape[-2])
            new_logits = probs_to_logits(action_probs).unsqueeze(-1)
            pd = self.action_dist.proba_distribution(action_logits=new_logits)
            # type(pd)._validate_args = False
            return pd
        else:
            pd = self.action_dist.proba_distribution(action_logits=action_logits)
            # type(pd)._validate_args = False
            return pd

    def evaluate_actions(self, obs, actions, action_masks=None):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, noise=True)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
